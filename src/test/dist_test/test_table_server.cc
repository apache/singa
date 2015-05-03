//  Copyright © 2014 Anh Dinh. All Rights Reserved.

#include "core/global-table.h"
#include "core/common.h"
#include "core/table.h"
#include "core/table_server.h"
#include "utils/global_context.h"
#include "utils/common.h"
#include <gflags/gflags.h>
#include "proto/model.pb.h"
#include "proto/common.pb.h"
#include "worker.h"
#include "coordinator.h"
#include "utils/common.h"
#include "utils/proto_helper.h"

#include <cmath>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>


/**
 * Test for table server access. The table is of type <VKey,int>
 */
DEFINE_bool(restore_mode, false, "restore from checkpoint file");
using namespace lapis;
using std::vector;

DEFINE_int32(checkpoint_frequency, 5000, "frequency for cp");
DEFINE_int32(checkpoint_after, 1, "cp after this steps");
DEFINE_string(par_mode, "hybrid",  "time training algorithm");
DEFINE_bool(restore, false, "restore from checkpoint file");

DEFINE_string(db_backend, "lmdb", "backend db");
DEFINE_string(system_conf, "examples/imagenet12/system.conf", "configuration file for node roles");
DEFINE_string(model_conf, "examples/imagenet12/model.conf", "DL model configuration file");
DEFINE_string(checkpoint_dir,"/data1/wangwei/lapis/","check point dir");
DEFINE_int32(threshold,1000000, "max # of parameters in a vector");
DEFINE_int32(iterations,5,"numer of get/put iterations");
DEFINE_int32(workers,2,"numer of workers doing get/put");
DECLARE_bool(checkpoint_enabled);


DECLARE_bool(checkpoint_enabled);

/**
 * Get and update handler for VKey.
 */
struct AnhUpdateHandler: BaseUpdateHandler<VKey, SGDValue> {
	bool Update(SGDValue *a, const SGDValue &b) {

		float * adptr = a->mutable_data()->mutable_value()->mutable_data();
		const float*bdptr = b.grad(0).value().data();
		for (int i = 0; i < b.grad(0).value_size(); i++)
			adptr[i] += bdptr[i];

		return true;
	}

	bool Get(const VKey k, const SGDValue &val, SGDValue *ret) {
		*ret = val;
		return true;
	}

	bool is_checkpointable(const VKey k, const SGDValue v) {
		return false; //always checkpoint
	}
};

typedef map<int, GlobalTable*> Map;
Map tables;
shared_ptr<NetworkThread> network;
shared_ptr<GlobalContext> context;
std::vector<ServerState*> server_states;
TableServer *table_server;

#define SIZE 16
int tuple_sizes[SIZE] = {27448736, 16777216, 4096000, 1327104, 884736, 884736, 614400,14112,4096,4096,1000,384,384,256,256,96};

/**
 * Initialize tables.
 */
void create_mem_table(int id, int num_shards){

	TableDescriptor *info = new TableDescriptor(id, num_shards);
	  info->key_marshal = new Marshal<VKey>();
	  info->value_marshal = new Marshal<SGDValue>();
	  info->sharder = new VKeySharder;
	  info->accum = new AnhUpdateHandler;
	  info->partition_factory = new typename SparseTable<VKey, SGDValue>::Factory;
	  auto table=new TypedGlobalTable<VKey, SGDValue>();
	  table->Init(info);
	  tables[id] = table;
}

/**
 * Coordinator assigns shards to processes.
 * @param id table ID.
 */
void coordinator_assign_tables(int id) {

	// wait for the servers to be up.
	for (int i = 0; i < context->num_procs(); i++) {
		RegisterWorkerRequest req;
		int src = 0;
		//  adding memory server.
		if (context->IsTableServer(i)) {
			VLOG(3)<< "Waiting for message from table server " << i;
			network->Read(MPI::ANY_SOURCE, MTYPE_REGISTER_WORKER, &req, &src);
			server_states.push_back(new ServerState(i));
		}
	}

	VLOG(3) << " All servers registered and started up. Ready to go";
	VLOG(3) << "num of shards" << tables[id]->num_shards() << " for table "	<< id;

	// assign table to shard in round roubin fashion.
	int server_idx = 0;
	for (int shard = 0; shard < tables[id]->num_shards(); ++shard) {
		ServerState &server = *server_states[server_idx];
		VLOG(3) << "Assigning table (" << id << "," << shard << ") to server "
				<< server_states[server_idx]->server_id;
		server.shard_id = shard;
		server.local_shards.insert(new TaskId(id, shard));
		server_idx = (server_idx + 1) % server_states.size();
	}
	ShardAssignmentRequest req;
	for (size_t i = 0; i < server_states.size(); ++i) {
		ServerState &server = *server_states[i];
		for (auto * task : server.local_shards) {
			ShardAssignment *s = req.add_assign();
			s->set_new_worker(server.server_id);
			s->set_table(task->table);
			s->set_shard(task->shard);
			//  update local tables
			GlobalTable *t = tables.at(task->table);
			t->get_partition_info(task->shard)->owner = server.server_id;
			delete task;
		}
	}

	network->SyncBroadcast(MTYPE_SHARD_ASSIGNMENT, MTYPE_SHARD_ASSIGNMENT_DONE,
			req);
	VLOG(3) << "done table assignment... ";
}


void table_init(){
	table_server = new TableServer();
	table_server->StartTableServer(tables);
	VLOG(3) << "table server started on process "<< NetworkThread::Get()->id();
}


/**
 * Coordinator loads data to the table.
 * @param size number of tuples.
 */
void coordinator_load_data() {
	auto table = static_cast<TypedGlobalTable<VKey, SGDValue>*>(tables[0]);
	for (int i = 0; i < SIZE; i++) {
		VKey key;
		SGDValue x;
		DAryProto *data = x.mutable_data();
		DAryProto *grad = x.add_grad();
		for (int j = 0; j < tuple_sizes[i]; j++) {
			data->add_value(j * 1.0f);
			grad->add_value(j * 1.0f);
		}
		key.set_key(i);
		table->put(key, x);
	}
	VLOG(3) << "Done loading " << SIZE << " tuples ...";
}

/**
 * Worker gets tuples from the server.
 * @param size number of tuples to be requested.
 */
void get() {
	auto table = static_cast<TypedGlobalTable<VKey,SGDValue>*>(tables[0]);
	SGDValue value;
	for (int i = 0; i < SIZE; i++) {
		VKey key;
		key.set_key(i);
		table->async_get(key, &value);
	}
	VLOG(3) << "Done sending get requests ...";

	for (int i = 0; i < SIZE; i++) {
		VKey key;
		while (!table->async_get_collect(&key, &value))
			Sleep(0.0001);
	}
}

/**
 * Worker updates tuples.
 */
void update() {
	auto table = static_cast<TypedGlobalTable<VKey, SGDValue>*>(tables[0]);
	for (int i = 0; i < SIZE; i++) {
		VKey key;
		key.set_key(i);

		SGDValue x;
		DAryProto *grad = x.add_grad();
		for (int j = 0; j < tuple_sizes[i]; j++)
			grad->add_value(j * 1.0f);

		table->update(key, x);
	}
	VLOG(3) << "Done updating " << SIZE << " tuples ...";
}


void worker_test_data() {
	//get(size);
	update();
	update();
	get();
	/*
	update(table, tuples);
	update(table, tuples);
	update(table, tuples);
	get(table, tuples);
	*/
}

/**
 * Shutdown the process.
 */
void shutdown() {
	if (context->AmICoordinator()) {
		EmptyMessage msg;
		for (int i = 0; i < context->num_procs() - 1; i++)
			network->Read(MPI::ANY_SOURCE, MTYPE_WORKER_END, &msg);
		EmptyMessage shutdown_msg;
		for (int i = 0; i < network->size() - 1; i++) {
			network->Send(i, MTYPE_SHUTDOWN, shutdown_msg);
		}
		//network->Flush();
		network->Shutdown();
	} else {
		//network->Flush();
		network->Send(context->num_procs() - 1, MTYPE_WORKER_END,
				EmptyMessage());
		EmptyMessage msg;
		network->Read(context->num_procs() - 1, MTYPE_SHUTDOWN, &msg);

		if (context->AmITableServer()){
			RequestDispatcher::Get()->PrintStats();
			table_server->ShutdownTableServer();
		}

		network->Shutdown();
	}
}

/**
 * Worker handle shard assignment from the coordinator.
 */
void HandleShardAssignment() {

	ShardAssignmentRequest shard_req;
	auto mpi = NetworkThread::Get();
	mpi->Read(GlobalContext::kCoordinator, MTYPE_SHARD_ASSIGNMENT, &shard_req);

	//  request read from coordinator
	for (int i = 0; i < shard_req.assign_size(); i++) {
		const ShardAssignment &a = shard_req.assign(i);
		GlobalTable *t = tables.at(a.table());
		t->get_partition_info(a.shard())->owner = a.new_worker();

		//if local shard, create check-point files
		if (FLAGS_checkpoint_enabled && t->is_local_shard(a.shard())) {
			string checkpoint_file = StringPrintf("%s/checkpoint_%d",
					FLAGS_checkpoint_dir.c_str(), a.shard());
			char hostname[256];
			gethostname(hostname, sizeof(hostname));

			FILE *tmp_file = fopen(checkpoint_file.c_str(), "r");
			if (tmp_file) { //exists -> open to reading and writing
				fclose(tmp_file);
				auto cp = t->checkpoint_files();

				if (FLAGS_restore_mode) { //open in read mode to restore, then close
					LogFile *file = new LogFile(checkpoint_file, "rw", 0);
					int table_size = file->read_latest_table_size();
					delete file;

					double start = Now();
					(*cp)[a.shard()] = new LogFile(checkpoint_file, "r",
							a.shard());
					t->Restore(a.shard());
					delete (*cp)[a.shard()];
					double end = Now();
					LOG(ERROR) << "restore time\t" << end - start << "\tfor\t"
							<< table_size << "\tthreshold\t" << FLAGS_threshold;
				}
				char hostname[256];
				gethostname(hostname, sizeof(hostname));
				(*cp)[a.shard()] = new LogFile(checkpoint_file, "a", a.shard());
			} else { // not exist -> open to writing first time
				auto cp = t->checkpoint_files();
				(*cp)[a.shard()] = new LogFile(checkpoint_file, "w", a.shard());
			}
		}
	}

	EmptyMessage empty;
	mpi->Send(GlobalContext::kCoordinator, MTYPE_SHARD_ASSIGNMENT_DONE, empty);
	VLOG(3) << "Done handling shard assignment ...";

}


int main(int argc, char **argv) {
	FLAGS_logtostderr = 1;
	int provided;
	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
	google::InitGoogleLogging(argv[0]);
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	context = GlobalContext::Get(FLAGS_system_conf);
	network = NetworkThread::Get();

	ModelProto model;
	ReadProtoFromTextFile(FLAGS_model_conf.c_str(), &model);

	create_mem_table(0, context->num_table_servers());

	if (context->AmICoordinator()) {
		coordinator_assign_tables(0);
		coordinator_load_data();
		network->barrier();
	} else {
		if (context->AmITableServer()) {
			table_init();
			HandleShardAssignment();
			network->barrier();
		} else {
			HandleShardAssignment();
			network->barrier();
			Sleep(1);
			VLOG(3) << "Worker cleared the barrier ...";
			worker_test_data();
		}
	}

	shutdown();
	return 0;
}


