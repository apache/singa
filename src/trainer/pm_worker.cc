#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include "gflags/gflags.h"
#include <glog/logging.h>
#include "proto/model.pb.h"
#include "trainer/pm_worker.h"
#include "mshadow/tensor.h"
#include "utils/cluster.h"


namespace singa{

void PMWorker::Setup(int group_id, int worker_id,
    shared_ptr<ParamShard> shard){
  group_id_=group_id;
  worker_id_=worker_id;
  shard_=shard;
}
int PMWorker::Sharding(int param_id){
  return param_id%Cluster::Get()->nservers_per_group();
}
/*
int PMWorker::Sharding(int param_id){
  static map<int, int> id2procs;
  if(id2procs.find(param_id)==id2procs.end()){
  auto cluster=Cluster::Get();
  int server_group=group_id_%cluster->nserver_groups();
  int nprocs_per_server_group=
    cluster->nservers_per_group()/cluster->nservers_per_procs();
  int procsid=server_group*nprocs_per_server_group+
    param_id%nprocs_per_server_group;
  procsid= cluster->server_worker_separate()?
    cluster->nworker_procs()+procsid:procsid;
  id2procs[param_id]=procsid;
  }
  return id2procs[param_id];
}
*/

Msg* PMWorker::Put(Msg** msg){
  return *msg;
}

Msg* PMWorker::Put(shared_ptr<Param> param, int step){
  param->set_version(step);
  // only owner can put shared parameter
  if(param->owner()<0||param->owner()==param->id()){
    Msg* msg= param->GenPutMsg(&step);
    msg->set_src(group_id_, worker_id_, kWorkerParam);
    msg->set_dst(group_id_/Cluster::Get()->nworker_groups_per_server_group(),
        Sharding(param->id()), kServer);
    msg->set_type(kPut);
    msg->set_target(param->id());
    return msg;
  }else
    return nullptr;
}

Msg* PMWorker::Get(Msg** msg){
  return *msg;
}

Msg* PMWorker::Get(shared_ptr<Param> param, int step){
  param->set_version(step);
  bool send=false;
  int id=param->id();
  shared_ptr<ParamCounter> entry=nullptr;
  if(param->owner()>=0){
    entry=shard_->at(id);
    entry->nGet++;
    send=entry->nGet/entry->nLocal==step;
  }
  if(param->owner()<0||send){
    Msg* msg=nullptr;
    if(param->owner()<0){
      msg=param->GenGetMsg(&step);
      msg->set_dst(group_id_/Cluster::Get()->nworker_groups_per_server_group(),
          Sharding(id), kServer);
    } else {
      msg=entry->param->GenGetMsg(&step);
      msg->set_dst(entry->owner_procs,kStub);
    }
    msg->set_src(group_id_, worker_id_, kWorkerParam);
    msg->set_type(kGet);
    msg->set_target(id);
    return msg;
  }else
    return nullptr;
}

Msg* PMWorker::Update(Msg** msg){
  return *msg;
}
Msg* PMWorker::Update(shared_ptr<Param> param, int step){
  param->set_version(step);
  bool send=false;
  int id=param->id();
  shared_ptr<ParamCounter> entry;
  if(param->owner()>=0){
    entry=shard_->at(param->id());
    entry->nGet++;
    send=entry->nGet/entry->nLocal==step;
    auto shape=mshadow::Shape1(param->size());
    mshadow::Tensor<mshadow::cpu,1> grad(param->mutable_cpu_grad(), shape);
    mshadow::Tensor<mshadow::cpu,1> agg(entry->param->mutable_cpu_grad(), shape);
    agg+=grad;
  }
  if(param->owner()<0||send){
    Msg* msg=nullptr;
    if(param->owner()<0){
      msg=param->GenUpdateMsg(&step);
      msg->set_dst(group_id_/Cluster::Get()->nworker_groups_per_server_group(),
          Sharding(id), kServer);
    } else {
      entry->param->GenUpdateMsg(&step);
      msg->set_dst(entry->owner_procs,kStub);
      memset(param->mutable_cpu_data(), 0, sizeof(float)*param->size());
    }
    msg->set_type(kUpdate);
    msg->set_target(id);
    msg->set_src(group_id_, worker_id_, kWorkerParam);
    return msg;
  }else
    return nullptr;
}

Msg* PMWorker::Collect(Msg** msg){
  int id=(*msg)->target();
  int type=(*msg)->type();
  auto pp=shard_->at(id)->param;
  if(type==kRGet){
    pp->ParseGetResponseMsg(msg);
  }else if(type==kRUpdate){
    pp->ParseUpdateResponseMsg(msg);
  }
  if(pp->owner()>=0){
    // forwarding to workers on other procs
  }
  delete (*msg);
  *msg=nullptr;
  return nullptr;
}

/*
//id is the global worker id
SingaClient::SingaClient(int global_id, Topology &topology, vector<string> &hosts) {
	//Read the config files and store endpoints
	id_ = global_id;

	int n_workers = hosts.size() - topology.nservers();
	int n_worker_groups = topology.nworker_groups();
	int group_size = n_workers/n_worker_groups;
	int server_group_size = topology.nservers()/topology.server_group_size();
	FLAGS_client_threads = topology.worker_threads();

	local_id_ = (id_-topology.nservers())%group_size;//local worker id.
	group_id_ = (id_-topology.nservers())/group_size;

	VLOG(3) << "Parsing client config for "<<hosts[id_];

	//connect to all server in the server group group_id_
	int start_server_idx = group_id_*server_group_size;
	int end_server_idx = start_server_idx+server_group_size;

	for (int i = start_server_idx; i < end_server_idx; i++) {
		char *neighbor_endpoint = (char*) malloc(256);
		sprintf(neighbor_endpoint, "tcp://%s:%d", hosts[i].c_str(), topology.port());
		neighbors_.push_back(neighbor_endpoint);
		VLOG(3) << "Worker neighbor (server): "<<neighbor_endpoint;
	}

	sprintf(backend_endpoint_, "inproc://singanus%d",id_);

	//Create shared paramshard
	param_shard_ = new ParamShard(id_,0);
}

void SingaClient::StartClient(){
	//Create and connect sockets to the server
	vector<void *> server_sockets;
	zctx_t *context = zctx_new();
	int nservers = neighbors_.size();
	int rc;
	for (int i=0; i<nservers; i++){
		void *socket = zsocket_new(context, ZMQ_DEALER);
		rc = zsocket_connect(socket, neighbors_[i]);
		VLOG(3) << "Connected to neighbor " <<neighbors_[i];
		assert(rc==0);
		server_sockets.push_back(socket);
	}

	//Create and bind backend socket
	void *backend = zsocket_new(context, ZMQ_ROUTER);
	rc = zsocket_bind(backend, backend_endpoint_);
	assert(rc==0);

	//Start client threads
	for (int i=0; i<FLAGS_client_threads; i++){
		void * socket = zthread_fork(context, ClientThread, this);
		zmsg_t *control_msg = zmsg_new();
		if (i==0 && local_id_==0)
			zmsg_pushstr(control_msg,POPULATE);
		else
			zmsg_pushstr(control_msg, WAIT);
		zmsg_send(&control_msg, socket);
	}

	//Star the message loop
	bool is_running = true;
	int nsockets= nservers+1;
	while (is_running) {
		zmq_pollitem_t items[nsockets];
		for (int i = 0; i < nsockets-1; i++)
			items[i] = {server_sockets[i], 0, ZMQ_POLLIN, 0};
		items[nsockets-1] = {backend, 0, ZMQ_POLLIN, 0};

		int rc = zmq_poll(items,nsockets,-1);
		if (rc<0) break;

		for (int i=0; i<nsockets-1; i++){
			if (items[i].revents & ZMQ_POLLIN){
				zmsg_t *msg = zmsg_recv(server_sockets[i]);
				if (!msg){
					is_running = false;
					break;
				}
				//forward to backend
				zmsg_send(&msg, backend);
			}
		}
		if (items[nsockets-1].revents & ZMQ_POLLIN){
			//compute serverId from paramId and forward to the socket
			zmsg_t *msg = zmsg_recv(backend);
			if (!msg) is_running=false;
			zframe_t *identity = zmsg_pop(msg);
			zframe_t *type = zmsg_pop(msg);
			int paramId;
			sscanf(zmsg_popstr(msg), "%d", &paramId);
			zmsg_pushstrf(msg,"%d",paramId);
			zmsg_prepend(msg,&type);
			zmsg_prepend(msg,&identity);
			zmsg_send(&msg, server_sockets[param_to_server_id(paramId)]);
		}
	}

	zsocket_destroy(context, backend);
	for (int i=0; i<nsockets-1; i++)
		zsocket_destroy(context, server_sockets[i]);
	zctx_destroy(&context);
}

vector<Param*> gen_random_params() {
	int size[] = { 1960000, 2500, 5000000, 2000, 3000000, 1500, 1500000, 1000, 500000, 500, 5000, 10 };
	vector<Param*> params;
	for (int i = 0; i < 12; i++) {
		ParamProto proto;
		proto.set_id(i);
		proto.set_init_method(ParamProto::kGaussain);
		Param* p = new Param();
		p->Setup(proto, vector<int> { size[i] }, 0);
		p->Init();
		params.push_back(p);
	}
	return params;
}

//simple mapping
int SingaClient::param_to_server_id(int paramId){
	return paramId % neighbors_.size();
}

void ClientThread(void *args, zctx_t *ctx, void *pipe){
	SingaClient *client = static_cast<SingaClient*>(args);

	//Create back-end socket and connect to the main thread
	void *backend = zsocket_new(ctx, ZMQ_DEALER);
	int rc = zsocket_connect(backend, client->backend_endpoint());
	assert(rc==0);
	//Create PMClient object
	PMClient *pmclient = new PMClient(client->id(), client->param_shard(), backend);

	//FOR TESTING ONLY. REMOVE THIS!
	//wait for control from main thread
	vector<Param*> params = gen_random_params();
	zmsg_t *control_msg = zmsg_recv(pipe);
	zframe_t *msg = zmsg_pop(control_msg);
	if (zframe_streq(msg,WAIT))
		zclock_sleep(2000); //2s
	else{
		for (int i=0; i<params.size(); i++){
			pmclient->Put(i, params[i]);
		}
		VLOG(3)<<"Done PUT requests for populating servers.";
		zclock_sleep(2000);
	}
	zframe_destroy(&msg);
	//END TESTING
	LOG(ERROR) << "Done putting";

	//first, get the params

	test_get(pmclient);
	test_collect(pmclient);


	int iterations = 1;
	while (iterations<=200){
		VLOG(3) << "Iteration "<<iterations;
		test_update(pmclient, params);
		test_collect(pmclient);
		iterations++;
	}

	zsocket_destroy(ctx, backend);
}

void test_get(PMClient *client){
	for (int i=0; i<12; i++){
		Param pm;
		int status = client->Get(i, &pm);
		assert(status==NON_LOCAL);
	}
}

void test_collect(PMClient *client){
	for (int i=0; i<12; i++){
		Param pm;
		int64_t start_time = zclock_time();
		while (!client->Collect(&pm))
			zclock_sleep(1);
		int64_t end_time = zclock_time();
		VLOG(3) << "Collected: " <<(end_time-start_time);
	}
}

void test_update(PMClient *client, vector<Param*> params){
	for (int i=0; i<params.size(); i++)
		client->Update(i, params[i]);
}
*/


} //namespace singa
