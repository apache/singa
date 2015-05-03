//  Copyright © 2014 Anh Dinh. All Rights Reserved.



#include "core/global-table.h"
#include "core/common.h"
#include "core/disk-table.h"
#include "core/table.h"
#include "core/table_server.h"
#include "utils/global_context.h"
#include <gflags/gflags.h>
#include "proto/model.pb.h"
#include "worker.h"
#include "coordinator.h"
#include "model_controller/myacc.h"
#include <cmath>

using namespace lapis;

DEFINE_bool(sync_update, false, "Synchronous put/update queue");
DEFINE_string(system_conf, "examples/imagenet12/system.conf", "configuration file for node roles");
DEFINE_string(model_conf, "examples/imagenet12/model.conf", "DL model configuration file");
DEFINE_int32(num_keys,10,"");

typedef map<int, GlobalTable*> Map;
Map tables;
shared_ptr<NetworkThread> network;
shared_ptr<GlobalContext> context;
std::vector<ServerState*> server_states;
TableServer *table_server;

void create_mem_table(int id, int num_shards){

	TableDescriptor *info = new TableDescriptor(id, num_shards);
	  info->key_marshal = new Marshal<int>();
	  info->value_marshal = new Marshal<int>();
	  info->sharder = new Sharding::Mod;
	  info->accum = new TestUpdater();
	  info->partition_factory = new typename SparseTable<int, int>::Factory;
	  auto table=new TypedGlobalTable<int, int>();
	  table->Init(info);
	  tables[id] = table;
}

void coordinator_assign_tables(int id){
	for (int i = 0; i < context->num_processes()-1; ++i) {
	    RegisterWorkerRequest req;
	    int src = 0;
	    network->Read(MPI::ANY_SOURCE, MTYPE_REGISTER_WORKER, &req, &src);
	    //  adding memory server.
	    if (context->IsTableServer(i)) {
	      server_states.push_back(new ServerState(i));
	    }
	  }
	  LOG(INFO) << " All servers registered and started up. Ready to go";
	  //  set itself as the current worker for the table
	  tables[id]->worker_id_ = network->id();

	  // memory servers are specified in global context. Round-robin assignment

	    VLOG(3)<<"num of shards"<<tables[id]->num_shards()<<" for table"<< id;

	    int server_idx = 0;
	    for (int shard = 0; shard < tables[id]->num_shards(); ++shard) {
	      ServerState &server = *server_states[server_idx];
	      LOG(INFO) << "Assigning table ("<<id<<","<<shard<<") to server "
	                <<server_states[server_idx]->server_id;

	      // TODO(Anh) may overwrite this field if #shards>#table_servers
	      server.shard_id = shard;
	      server.local_shards.insert(new TaskId(id, shard));
	      server_idx = (server_idx + 1) % server_states.size();
	    }

	  VLOG(3)<<"table assignment";
	  //  then send table assignment
	  ShardAssignmentRequest req;
	  for (size_t i = 0; i < server_states.size(); ++i) {
	    ServerState &server = *server_states[i];
	    for (auto * task: server.local_shards) {
	      ShardAssignment *s  = req.add_assign();
	      s->set_new_worker(server.server_id);
	      s->set_table(task->table);
	      s->set_shard(task->shard);
	      //  update local tables
	      CHECK(tables.find(task->table)!=tables.end());
	      GlobalTable *t = tables.at(task->table);
	      t->get_partition_info(task->shard)->owner = server.server_id;
	      delete task;
	    }
	  }
	  VLOG(3)<<"finish table assignment, req size "<<req.assign_size();
	  network->SyncBroadcast(MTYPE_SHARD_ASSIGNMENT, MTYPE_SHARD_ASSIGNMENT_DONE, req);
	  VLOG(3)<<"finish table server init";
}

void worker_table_init(){
	table_server = new TableServer();
	table_server->StartTableServer(tables);
	VLOG(3) << "done starting table server";
}


void coordinator_load_data(){
	auto table = static_cast<TypedGlobalTable<int,int>*>(tables[0]);
	for (int i = 1; i<=FLAGS_num_keys; i++){
		table->put(i,i);
	}
	VLOG(3) << "Loaded data successfully ...";
}

void worker_test_data(){
	auto table = static_cast<TypedGlobalTable<int,int>*>(tables[0]);
	for (int i=1; i<=FLAGS_num_keys; i++)
		VLOG(3) << StringPrintf("Worker %d got (%d,%d)", NetworkThread::Get()->id(), i, table->get(i));


	for (int j = 0; j < 2; j++) {
		for (int i = 1; i <= FLAGS_num_keys; i++)
			table->update(i, i);

		for (int i = 1; i <= FLAGS_num_keys; i++)
			VLOG(3)
					<< StringPrintf("Worker %d got (%d,%d)",
							NetworkThread::Get()->id(), i, table->get(i));
	}
/*
	for (int i = 1; i <= FLAGS_num_keys; i++)
				VLOG(3)
						<< StringPrintf("Worker %d got (%d,%d)",

							NetworkThread::Get()->id(), i, table->get(i));
*/
}

void shutdown(){
	if (context->AmICoordinator()){
		VLOG(3) << "Coordinator is shutting down ...";
		EmptyMessage msg;
		for (int i=0; i<context->num_processes()-1; i++)
			network->Read(MPI::ANY_SOURCE, MTYPE_WORKER_END, &msg);
		 EmptyMessage shutdown_msg;
		  for (int i = 0; i < network->size() - 1; i++) {
		    network->Send(i, MTYPE_WORKER_SHUTDOWN, shutdown_msg);
		  }
		  network->Flush();
		  network->Shutdown();
	}
	else{
		VLOG(3) << "Worker " << network->id() << " is shutting down ...";
	  network->Flush();
	  VLOG(3) << "Done flushing the network thread";
	  network->Send(GlobalContext::kCoordinatorRank, MTYPE_WORKER_END, EmptyMessage());
	  EmptyMessage msg;
	  network->Read(GlobalContext::kCoordinatorRank, MTYPE_WORKER_SHUTDOWN, &msg);
	  VLOG(3) << "Worker received MTYPE_WORKER_SHUTDOWN";
	  table_server->ShutdownTableServer();
	  VLOG(3) << "Flushing node " << network->id();
	  network->Shutdown();
	}
}


int main(int argc, char **argv) {
	FLAGS_logtostderr = 1;
	google::InitGoogleLogging(argv[0]);
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	context = GlobalContext::Get(FLAGS_system_conf, FLAGS_model_conf);
	network = NetworkThread::Get();
	VLOG(3) << "*** testing memory servers, with "
			<< context->num_table_servers() << " servers";
	create_mem_table(0,context->num_table_servers());

	if (context->AmICoordinator()){
		coordinator_assign_tables(0);
		coordinator_load_data();
		network->barrier();
	}
	else{
		worker_table_init();
		network->barrier();
		VLOG(3) << "passed the barrier";
		//Sleep(1);
		worker_test_data();
	}

	shutdown();
	return 0;
}


