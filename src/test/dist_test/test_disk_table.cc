//  Copyright © 2014 Anh Dinh. All Rights Reserved.
//  main class for testing distributed memory layer
//
//  the command to run this should be:
//		mpirun -hostfile <host> -bycore -nooversubscribe
//				-n <num_servers> test -sync_update


#include "core/global-table.h"
#include "core/common.h"
#include "core/disk-table.h"
#include "core/table.h"
#include "core/table_server.h"
#include "utils/global_context.h"
#include <gflags/gflags.h>
#include "proto/model.pb.h"
#include "worker.h"
#include <cmath>

DEFINE_int32(record_size,100, "# elements per float vector");
DECLARE_int32(block_size);
DEFINE_int32(table_size, 1000, "# records per table");
DEFINE_string(system_conf, "examples/imagenet12/system.conf", "configuration file for node roles");
DEFINE_string(model_conf, "examples/imagenet12/model.conf", "DL model configuration file");
DEFINE_bool(is_testing_put,true, "data put vs. data get");
DECLARE_int32(debug_index);
DECLARE_int32(table_buffer);
using namespace lapis;

typedef map<int, GlobalTable*> Map;
Map tables;

//  put random message to the pointers
void create_random_message(FloatVector* message, const int count){
	for (int i=0; i<FLAGS_record_size; i++){
		message->add_data(count*FLAGS_record_size+i);
	}
}

void create_disk_table(int id){
	DiskTableDescriptor *info = new DiskTableDescriptor(id, "disk_test",
			FLAGS_block_size);
	info->key_marshal = new Marshal<int>();
	info->value_marshal = new Marshal<FloatVector>();
	tables[id] = new TypedDiskTable<int,FloatVector>(info);
}


//  if testing put, write and send data. Else do nothing
void run_coordinator(shared_ptr<NetworkThread> network, int tid){
	// wait for wokers to be up
	RegisterWorkerRequest req;
	for (int i=0; i<network->size()-1; i++)
		network->Read(MPI::ANY_SOURCE, MTYPE_REGISTER_WORKER, &req);

	// put data in
	TypedDiskTable<int, FloatVector>* table = static_cast<TypedDiskTable<int,
			FloatVector>*>(tables[tid]);

	//  if testing put()
	if (FLAGS_is_testing_put) {
		int count = 0;
		for (int i = 0; i < FLAGS_table_size; i++) {
			FloatVector message;
			create_random_message(&message, i);
			table->put(i, message);
			count += message.ByteSize();
		}
		table->finish_put();
	}

	VLOG(3) << "Coordinator about to shut down";
	for (int i=0; i<network->size()-1; i++){
		EmptyMessage end_msg;
		network->Read(i,MTYPE_WORKER_END, &end_msg);
	}

	EmptyMessage shutdown_msg;
	for (int i = 0; i < network->size() - 1; i++) {
		network->Send(i, MTYPE_WORKER_SHUTDOWN, shutdown_msg);
	}
	network->Flush();
	network->Shutdown();
	table->PrintStats();

	if (FLAGS_is_testing_put) {
		int sub_blocks = ceil(((double) FLAGS_table_size / FLAGS_table_buffer));
		CHECK_EQ(table->stats()["total sub block sent"], sub_blocks);
		CHECK_EQ(table->stats()["total record sent"], FLAGS_table_size);
		VLOG(3) << "test coordinator sending: successful";
	}

}

//  if testing put(), do nothing. Else read() until done()
void run_worker(shared_ptr<NetworkThread> network, int tid){
	TableServer* ts = new TableServer();
	ts->StartTableServer(tables);

	// put data in
	TypedDiskTable<int, FloatVector>* table = static_cast<TypedDiskTable<int,
			FloatVector>*>(tables[tid]);
	double total_read = 0;
	if (!FLAGS_is_testing_put){
		VLOG(3) << "testing read from table ...";
		table->Load();
		while (!table->done()){
			int k;
			FloatVector v;
			table->get(&k,&v);
			table->Next();
			total_read++;
		}

		int k;
		FloatVector v;
		table->get(&k, &v);
		total_read++;
	}

	int size = network->size();

	network->Flush();
	network->Send(GlobalContext::kCoordinatorRank, MTYPE_WORKER_END,
			EmptyMessage());
	EmptyMessage msg;

	int src = 0;
	network->Read(GlobalContext::kCoordinatorRank, MTYPE_WORKER_SHUTDOWN, &msg,
			&src);
	network->Flush();
	network->Shutdown();

	Stats stats =
			(static_cast<TypedDiskTable<int, FloatVector>*>(tables[0]))->stats();

	if (FLAGS_is_testing_put) {
		int sub_blocks = ceil(((double) FLAGS_table_size / FLAGS_table_buffer));
		if (size == 2) {
			CHECK_EQ(stats["total sub block received"], sub_blocks);
			CHECK_EQ(stats["total record stored"], FLAGS_table_size);
		}
		VLOG(3) << "test table-server writing: successful";
		VLOG(3) << "number of sub blocks = " << sub_blocks;
		VLOG(3) << "total data stored = " << stats["total byte stored"];
	}
	else{
		if (size==2)
			CHECK_EQ(stats["total record read"], FLAGS_table_size);
		VLOG(3) << "test table-server reading: successful";
		VLOG(3) << "read bandwidth = "
				<< (stats["total byte read"]
						/ (stats["last byte read"] - stats["first byte read"]));
		//VLOG(3) << "total number of record read = " << stats["total record read"];
	}

	network->PrintStats();
	static_cast<TypedDiskTable<int, FloatVector>*>(tables[0])->PrintStats();
}

//  check all the records have been stored to disk
int test_disk(int tid) {
	// Init GlobalContext
	auto gc = lapis::GlobalContext::Get(FLAGS_system_conf, FLAGS_model_conf);
	//start network thread
	shared_ptr<NetworkThread> network = NetworkThread::Get();

	if (network->id() == network->size() - 1)
		run_coordinator(network, tid);
	else
		run_worker(network,tid);
	return 0;
}

// for debugging use
//#ifndef FLAGS_v
//  DEFINE_int32(v, 3, "vlog controller");
//#endif

int main(int argc, char **argv) {
	FLAGS_logtostderr = 1;
	google::InitGoogleLogging(argv[0]);
	gflags::ParseCommandLineFlags(&argc, &argv, true);
	create_disk_table(0);
	return test_disk(0);
}


