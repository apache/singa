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


DEFINE_bool(restore_mode, false, "restore from checkpoint file");
using namespace lapis;
using std::vector;

//DEFINE_bool(sync_update, false, "Synchronous put/update queue");
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

#ifndef FLAGS_v
  DEFINE_int32(v, 3, "vlog controller");
#endif


struct AnhUpdateHandler: BaseUpdateHandler<VKey,SGDValue>{
	bool Update(SGDValue *a, const SGDValue &b){
    float * adptr=a->mutable_data()->mutable_value()->mutable_data();
    const float*bdptr=b.grad(0).value().data();
    for(int i=0;i<b.grad(0).value_size();i++)
      adptr[i]+=bdptr[i];
		return true;
	}

  bool Get(const VKey k, const SGDValue &val, SGDValue *ret){
      *ret = val;
      return true;
  }

  bool is_checkpointable(const VKey k, const SGDValue v){
  	return true; //always checkpoint
  }
};

typedef map<int, GlobalTable*> Map;
Map tables;
shared_ptr<NetworkThread> network;
shared_ptr<GlobalContext> context;
std::vector<ServerState*> server_states;
TableServer *table_server;
TableDelegate *delegate;
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

void coordinator_assign_tables(int id){
	for (int i = 0; i < context->num_procs() 	; ++i) {
	    RegisterWorkerRequest req;
	    int src = 0;
	    //  adding memory server.
	    if (context->IsTableServer(i)) {
	      network->Read(MPI::ANY_SOURCE, MTYPE_REGISTER_WORKER, &req, &src);
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
void coordinator_load_data(const vector<int>& tuples){
  auto table = static_cast<TypedGlobalTable<VKey,SGDValue>*>(tables[0]);

  int nservers=context->num_table_servers();
  int keyid=0;
  if (!FLAGS_restore_mode){
    for(auto tuple: tuples){
      for(int offset=0;offset<tuple;){
        SGDValue x;
        DAryProto *data=x.mutable_data();
        DAryProto *grad=x.add_grad();
        for(int i=0;i <std::min(FLAGS_threshold, tuple-offset);i++){
          data->add_value(i*1.0f);
          grad->add_value(i*1.0f);
        }
        offset+=data->value_size();
        VKey key;
        key.set_key(keyid++);
        table->put(key,x);
      }
    }
    LOG(ERROR)<<"put "<<keyid<<" tuples";
  }

  /*
	LogFile *file = new LogFile("/data1/wangwei/lapis/checkpoint_0","rw",0);
	VLOG(3) << "Loaded table " << file->file_name();
	string k,v;
	int table_size = file->read_latest_table_size();
	VLOG(3) << "table size = " << table_size;
	for (int i=0; i<table_size; i++){
		int tmp;
		file->previous_entry(&k, &v, &tmp);
		int *key = reinterpret_cast<int *>((char*)&k[0]);
		int *val = reinterpret_cast<int *>((char*)&v[0]);
		VLOG(3) << "k = " << *key << " val = " << *val;
	}
	delete file;
  */

	/*
	for (int i=0; i<num_keys; i++){
		table->put(i,0); //loaded again
	}*/
	VLOG(3) << "Coordinator done loading ..., from process "<<NetworkThread::Get()->id();
}

void get(TypedGlobalTable<VKey,SGDValue>* table, const vector<int>& tuples){
  SGDValue v;
  int num_keys=0;
  for(auto tuple: tuples){
    num_keys+=tuple/FLAGS_threshold+(tuple%FLAGS_threshold!=0);
  }
  LOG(ERROR)<<"getting "<<num_keys<<" tuples";

  for (int i=0; i<num_keys; i++){
    VKey key;
    key.set_key(i);
    table->async_get(key, &v);
  }


  int key=0;
  SGDValue val;

  LOG(INFO)<<"start collect key";
  for (int i=0; i<num_keys; i++){
    VKey key;
    while(!table->async_get_collect(&key, &val))
      Sleep(0.001);
    //LOG(INFO)<<"collect key "<<key<<" with val "<<val;
  }
}

void update(TypedGlobalTable<VKey,SGDValue>* table, const vector<int>& tuples){
  if(NetworkThread::Get()->id()==0)
    sleep(2);
  LOG(INFO)<<"start update";
  int keyid=0;
  for(auto tuple: tuples){
    for(int offset=0;offset<tuple;){
      SGDValue x;
      DAryProto *grad=x.add_grad();
      for(int i=0;i <std::min(FLAGS_threshold, tuple-offset);i++){
        grad->add_value(i*1.0f);
      }
      offset+=grad->value_size();
      VKey key;
      key.set_key(keyid++);
      table->update(key,x);
    }
  }
  LOG(ERROR)<<"updated "<<keyid<<" tuples";
}

void worker_test_data(const vector<int>& tuples){
  auto table = static_cast<TypedGlobalTable<VKey,SGDValue>*>(tables[0]);

  get(table, tuples);
  update(table, tuples);
  update(table, tuples);
  update(table, tuples);
  get(table, tuples);
}

void shutdown(){
	if (context->AmICoordinator()){
		EmptyMessage msg;
		for (int i=0; i<context->num_procs()-1; i++)
			network->Read(MPI::ANY_SOURCE, MTYPE_WORKER_END, &msg);
		 EmptyMessage shutdown_msg;
		  for (int i = 0; i < network->size() - 1; i++) {
		    network->Send(i, MTYPE_SHUTDOWN, shutdown_msg);
		  }
		  network->Flush();
		  network->Shutdown();
	}
	else{
	  network->Flush();

	  network->Send(context->num_procs()-1, MTYPE_WORKER_END, EmptyMessage());

	  EmptyMessage msg;

	  network->Read(context->num_procs()-1, MTYPE_SHUTDOWN, &msg);

	  if (context->AmITableServer())
		  table_server->ShutdownTableServer();

	  network->Shutdown();
	}
}

void HandleShardAssignment() {

  ShardAssignmentRequest shard_req;
  auto mpi=NetworkThread::Get();
  mpi->Read(GlobalContext::kCoordinator, MTYPE_SHARD_ASSIGNMENT, &shard_req);
  //  request read from coordinator
  for (int i = 0; i < shard_req.assign_size(); i++) {
    const ShardAssignment &a = shard_req.assign(i);
    GlobalTable *t = tables.at(a.table());
    t->get_partition_info(a.shard())->owner = a.new_worker();


    //if local shard, create check-point files
    if (FLAGS_checkpoint_enabled && t->is_local_shard(a.shard())){
      string checkpoint_file = StringPrintf("%s/checkpoint_%d",FLAGS_checkpoint_dir.c_str(), a.shard());
        char hostname[256];
        gethostname(hostname, sizeof(hostname));
        VLOG(3) << "try to open for writing *****"<<checkpoint_file<<" "<<string(hostname);

      FILE *tmp_file = fopen(checkpoint_file.c_str(), "r");
      if (tmp_file){//exists -> open to reading and writing
        fclose(tmp_file);
        auto cp = t->checkpoint_files();

        if (FLAGS_restore_mode){//open in read mode to restore, then close
          LogFile *file = new LogFile(checkpoint_file,"rw",0);
          VLOG(3) << "Loaded table " << file->file_name();
          int table_size = file->read_latest_table_size();
          delete file;

          double start=Now();
          VLOG(3) << "Open checkpoint file to restore";
          (*cp)[a.shard()] = new LogFile(checkpoint_file,"r",a.shard());
          t->Restore(a.shard());
          delete (*cp)[a.shard()];
          double end=Now();
          LOG(ERROR)<<"restore time\t"<<end-start<< "\tfor\t"
            <<table_size<<"\tthreshold\t"<<FLAGS_threshold;
        }
        char hostname[256];
        gethostname(hostname, sizeof(hostname));
        VLOG(3) << "open for writing *****"<<checkpoint_file<<" "<<string(hostname);



        VLOG(3) << "Open checkpoint file for writing";
        (*cp)[a.shard()] = new LogFile(checkpoint_file,"a",a.shard());
      }
      else{// not exist -> open to writing first time
        auto cp = t->checkpoint_files();
        (*cp)[a.shard()] = new LogFile(checkpoint_file,"w",a.shard());
        VLOG(3) << "Added to new checkpoint files for shard "<< a.shard();
      }

    }


  }
  EmptyMessage empty;
  mpi->Send(GlobalContext::kCoordinator, MTYPE_SHARD_ASSIGNMENT_DONE, empty);
  VLOG(3)<<"finish handle shard assignment **";

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

	create_mem_table(0,context->num_table_servers());

  vector<int> tuple_size{37448736, 16777216, 4096000, 1327104, 884736, 884736, 614400,14112,4096,4096,1000,384,384,256,256,96};
  /*
  vector<int> tuples;
  for(int i=0;i<3;i++){
    for(int j=0;j<FLAGS_workers;j++)
      tuples.push_back(tuple_size[i]/FLAGS_workers);
  }
  for(int i=3;i<tuple_size.size();i++)
    tuples.push_back(tuple_size[i]);
    */

	if (context->AmICoordinator()){
		VLOG(3) << "Coordinator process rank = " << NetworkThread::Get()->id();
		coordinator_assign_tables(0);
		coordinator_load_data(tuple_size);

		network->barrier();
	}
	else{
		if (context->AmITableServer()){
			worker_table_init();
			HandleShardAssignment();
			network->barrier();
		}
		else{
			VLOG(3) << "Inside worker, waiting for assignemtn";
			HandleShardAssignment();
			network->barrier();
      if(!FLAGS_restore_mode)
        worker_test_data(tuple_size);
		}
	}
	shutdown();


	VLOG(3) << "Done ...";
	return 0;
}


