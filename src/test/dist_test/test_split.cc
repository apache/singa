//  Copyright © 2014 Anh Dinh. All Rights Reserved.


//  Testing the unbalance in spliting parameter vectors.

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
//#include "model_controller/myacc.h"
#include "utils/common.h"

#include <cmath>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>

using namespace lapis;
using std::vector;

//DEFINE_bool(sync_update, false, "Synchronous put/update queue");
DEFINE_string(system_conf, "examples/imagenet12/system.conf", "configuration file for node roles");
DEFINE_string(model_conf, "examples/imagenet12/model.conf", "DL model configuration file");
DEFINE_int64(threshold,1000000, "max # of parameters in a vector");
DEFINE_int32(iterations,5,"numer of get/put iterations");
DEFINE_int32(workers,2,"numer of workers doing get/put");
#ifndef FLAGS_v
  DEFINE_int32(v, 3, "vlog controller");
#endif

typedef map<int, GlobalTable*> Map;
Map tables;
shared_ptr<NetworkThread> network;
shared_ptr<GlobalContext> context;
std::vector<ServerState*> server_states;
TableServer *table_server;

FloatVector large_msg, small_msg;
const int SIZE=16;

long sizes[] = { 37448736, 16777216, 4096000, 1327104, 884736, 884736, 614400,
		14112, 4096, 4096, 1000, 384, 384, 256, 256, 96 };

vector<FloatVector*> value_msg;

int num_keys;

// create large and small messages
void init_messages(){
	num_keys = 0;
  long nservers=context->num_table_servers();
	for (int i=0; i<SIZE; i++){
		int total=0;
    int threshold=std::max(FLAGS_threshold,0l);//, sizes[i]/nservers);
    VLOG(3)<<"worker: "<<threshold;
		while (total<sizes[i]){
			FloatVector* fv = new FloatVector();
			for (int j=0; j+total<sizes[i] && j<threshold; j++)
				fv->add_data(static_cast<float>(rand())/static_cast<float>(RAND_MAX));
			value_msg.push_back(fv);
			total+=threshold;
			num_keys++;
		}
	}
}

void create_mem_table(int id, int num_shards){

	TableDescriptor *info = new TableDescriptor(id, num_shards);
	  info->key_marshal = new Marshal<int>();
	  info->value_marshal = new Marshal<FloatVector>();
	  info->sharder = new Sharding::Mod;
	  info->accum = new MyAcc();
	  info->partition_factory = new typename SparseTable<int, FloatVector>::Factory;
	  auto table=new TypedGlobalTable<int, FloatVector>();
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

double random_double(){
	return static_cast<double>(rand())/static_cast<double>(RAND_MAX);
}

// popular table with random large or small messages.
// the message distribution specified in FLAGS_large_precentage
void coordinator_load_data(){
	auto table = static_cast<TypedGlobalTable<int,FloatVector>*>(tables[0]);

	num_keys = 0;
  int nservers=context->num_table_servers();
	for (int i = 0; i < SIZE; i++) {
		int total = 0;
    int threshold=std::max(FLAGS_threshold,0l);//  sizes[i]/nservers);
    while (total < sizes[i]) {
      FloatVector* fv = new FloatVector();
      for (int j = 0; j + total < sizes[i] && j < threshold; j++)
        fv->add_data(
            static_cast<float>(rand())
            / static_cast<float>(RAND_MAX));
      table->put(num_keys,*fv);
      total += threshold;
      num_keys++;
    }
	}
	VLOG(3) << "Loaded data successfully ... " << num_keys << " messages";
}

void get(TypedGlobalTable<int,FloatVector>* table, ofstream &latency){
	double start , end;
  StateQueue<int> state(num_keys);
  FloatVector v;
  /*
	for (int i=0; i<num_keys; i++){
    start = Now();
    table->get(i);
    end=Now();
    latency << "get: " << (end - start) << endl;
  }
  */
  start=Now();
	for (int i=0; i<num_keys; i++){
    if(table->async_get(i, &v))
      state.Invalid(i);
	}
  latency << "send get: " << (Now() - start) << endl;
  start=Now();
  while(state.HasValid()){
    int key=state.Next();
    if(table->async_get_collect(&key, &v))
      state.Invalid(key);
    sleep(0.001);
  }
  latency << "collect get: " << (Now() - start) << endl;
}

void update(TypedGlobalTable<int,FloatVector>* table, ofstream &latency){
	double start, end;
	for (int i=0; i<num_keys; i++){
		start = Now();
		table->update(i,*value_msg[i]);
    end=Now();
		latency << "update: " << (end - start) << endl;
	}
}

void worker_test_data(){
	init_messages();
	auto table = static_cast<TypedGlobalTable<int,FloatVector>*>(tables[0]);

	ofstream latency(StringPrintf("latency_%d",NetworkThread::Get()->id()));
	ofstream throughput(StringPrintf("throughput_%d", NetworkThread::Get()->id()));
	double start, end;
	for (int i=0; i<FLAGS_iterations; i++){
		start = Now();
		get(table, latency);
    end=Now();
		throughput << "get: " << (end - start) << " over " << num_keys << " ops " << endl;
		start = Now();
		update(table, latency);
    end=Now();
		throughput << "update: " << (end - start) << " over " << num_keys << " ops " << endl;
    sleep(10);
	}
	latency.close();
	throughput.close();

}

void print_table_stats(){
	auto table = static_cast<TypedGlobalTable<int,FloatVector>*>(tables[0]);
	ofstream log_file(StringPrintf("log_variance_%d", NetworkThread::Get()->id()));
	log_file << "table size at process "<< NetworkThread::Get()->id()<<" = " << table->stats()["TABLE_SIZE"] << endl;
	log_file.close();
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

  LOG(INFO)<<"threshold: "<<FLAGS_threshold<<" nworkers: "<<FLAGS_workers;
	if (context->AmICoordinator()){
		coordinator_assign_tables(0);
		coordinator_load_data();
		network->barrier();
	}
	else{
		worker_table_init();
		network->barrier();
		VLOG(3) << "passed the barrier";
		print_table_stats();

		//Sleep(1);
    if(network->id()<FLAGS_workers)
      worker_test_data();
	}

	shutdown();
	return 0;
}


