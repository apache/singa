#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "server.h"
#include "proto/worker.pb.h"
#include "utils/network_service.h"
#include "core/common.h"
#include "core/network_queue.h"
#include "proto/model.pb.h"
#include "proto/common.pb.h"
#include "utils/global_context.h"

/**
 * @file test_tuple.cc
 *
 * Test performance of TableServer put/get/update operations.
 */
DECLARE_double(sleep_time);

using namespace lapis;
using namespace std;
using std::vector;

#define NKEYS 1000
#define TUPLE_SIZE 50000000

#ifndef FLAGS_v
  DEFINE_int32(v, 3, "vlog controller");
#endif


#define SIZE 16
#define THRESHOLD 500000
int tuple_sizes[SIZE] = {37448736, 16777216, 4096000, 1327104, 884736, 884736, 614400,14112,4096,4096,1000,384,384,256,256,96};
vector<int> valsizes;
int collect_size;
int num_tuples;

void Put(int tid, int size, int version) {
	RequestBase request;
	request.set_table(0);
	request.set_source(NetworkService::Get()->id());
	PutRequest *put_req = request.MutableExtension(PutRequest::name);
	int shard = tid % GlobalContext::Get()->num_servers();
	put_req->set_shard(shard);
	TableData *tuple = put_req->mutable_data();

	TKey* key = tuple->mutable_key();
	TVal* val = tuple->mutable_value();

	key->set_id(tid);
	key->set_version(version);

	DAryProto *data = val->mutable_data();
	for (int i = 0; i < size; i++){
		data->add_value(0.0f);
	}

	// TODO check the msg type
	NetworkService::Get()->Send(shard, MTYPE_REQUEST, request);
}

void Update(int tid, int size, int version) {
	RequestBase request;
	request.set_table(0);
	request.set_source(NetworkService::Get()->id());
	UpdateRequest *update_req = request.MutableExtension(UpdateRequest::name);
	int shard = tid % GlobalContext::Get()->num_servers();
	update_req->set_shard(shard);
	TableData *tuple = update_req->mutable_data();

	TKey* key = tuple->mutable_key();
	TVal* val = tuple->mutable_value();

	key->set_id(tid);
	key->set_version(version);

	DAryProto *data = val->mutable_grad();
	for (int i = 0; i < size; i++)
		data->add_value(1.0f);
	// TODO check the msg type
	NetworkService::Get()->Send(shard, MTYPE_REQUEST, request);
}

void print_result(TableData *data){
	TKey *key = data->mutable_key();
	TVal *val = data->mutable_value();
	int k = key->id();
	VLOG(3) << "key = " << k;
	string s;
	for (int i=0; i<TUPLE_SIZE; i++)
		s.append(to_string(val->mutable_data()->value(i))).append(" ");
	VLOG(3) << "val = " <<s;
}

void AsyncGet(int tid, int version) {
	RequestBase request;
	request.set_table(0);
	request.set_source(GlobalContext::Get()->rank()); //NetworkService::Get()->id());
	GetRequest *get_req = request.MutableExtension(GetRequest::name);
	int shard = tid % GlobalContext::Get()->num_servers();
	get_req->set_shard(shard);

	TKey *key = get_req->mutable_key();
	key->set_id(tid);
	key->set_version(version);
	NetworkService::Get()->Send(shard, MTYPE_REQUEST, request);

}

void Collect(){
	int count = collect_size;
	double start_collect = Now();
	while (count){
		while (true) {
				Message *resp = NetworkService::Get()->Receive();
				if (!resp)
					Sleep(FLAGS_sleep_time);
				else{
					delete resp;
					break;
				}
			}
		count--;
	}
	double end_collect = Now();
	VLOG(3) << "Collected " << collect_size << " tuples in " << (end_collect-start_collect);
}

/**
 * Workers wait for the barrier, then one of them send SHUTDOWN message
 * to all table servers.
 */
void worker_send_shutdown(int id){
	auto gc = lapis::GlobalContext::Get();
	NetworkService *network_service_ = NetworkService::Get().get();
	MPI_Barrier(gc->workergroup_comm());
	if (gc->rank()==id){
		for (int i=0; i<gc->num_procs(); i++){
			if (gc->IsTableServer(i)){
				EmptyMessage msg;
				network_service_->Send(i, MTYPE_SHUTDOWN,msg);
			}
		}
	}
}

/**
 * One worker with the specific ID puts, others wait.
 */
void worker_load_data(int id){
	auto gc = lapis::GlobalContext::Get();
	for (int i = 0; i < SIZE; i++) {
		int m = tuple_sizes[i];
		if (m < THRESHOLD)
			valsizes.push_back(m);
		else {
			for (int j = 0; j < m / THRESHOLD; j++)
				valsizes.push_back(THRESHOLD);
			if (m % THRESHOLD)
				valsizes.push_back(m%THRESHOLD);
		}
	}
	num_tuples = (int)valsizes.size();
	collect_size = 0;
	for (int i=0; i<num_tuples; i++)
		if (i%gc->group_size()==gc->worker_id())
			collect_size++;

	if (gc->rank()==id){
		for (size_t i=0; i<valsizes.size(); i++)
			Put(i,valsizes[i],0);
		VLOG(3) << "Done loading data, num_keys = "<<valsizes.size() << " process " << id;
	}
	VLOG(3) << "Collect size = " << collect_size;
	MPI_Barrier(gc->workergroup_comm());
}

void worker_update_data() {
	auto gc = lapis::GlobalContext::Get();
	for (int i = 0; i < num_tuples; i++)
		if (i%gc->group_size()==gc->worker_id())
			Update(i,valsizes[i],0);

	VLOG(3) << "Done update ... for "<<collect_size << " tuples ";
}

/*
 * Async get.
 */
void worker_get_data(){
	auto gc = lapis::GlobalContext::Get();
	for (int i=0; i<num_tuples; i++)
		if (i%gc->group_size()==gc->worker_id())
			AsyncGet(i,0);
	Collect();
	VLOG(3) << "Done collect ...";
}

void start_network_service_for_worker(){
	NetworkService *network_service_ = NetworkService::Get().get();
	network_service_->Init(GlobalContext::Get()->rank(), Network::Get().get(), new SimpleQueue());
	network_service_->StartNetworkService();
}

int main(int argc, char **argv) {
	google::InitGoogleLogging(argv[0]);
	gflags::ParseCommandLineFlags(&argc, &argv, true);

	int provided;


	MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);


	FLAGS_logtostderr = 1;


	// Init GlobalContext
	Cluster cluster;
	cluster.set_server_start(0);
	cluster.set_server_end(8);
	cluster.set_worker_start(8);
	cluster.set_worker_end(24);
	cluster.set_group_size(8);
	cluster.set_data_folder("/data1/wangwei/lapis");

	auto gc = lapis::GlobalContext::Get(cluster);

	// worker or table server
	if (gc->AmITableServer()) {
		lapis::TableServer server;
		SGDProto sgd;
		sgd.set_learning_rate(0.01);
		sgd.set_momentum(0.9);
		sgd.set_weight_decay(0.1);
		sgd.set_gamma(0.5);
		sgd.set_learning_rate_change_steps(1);
		server.Start(sgd);
	} else {
		start_network_service_for_worker();
		worker_load_data(cluster.worker_start());
		for (int i=0; i<10; i++){
			worker_update_data();
			worker_get_data();
		}
		worker_send_shutdown(cluster.worker_start());
		NetworkService::Get()->Shutdown();
	}
	gc->Finalize();
	MPI_Finalize();
	VLOG(3) << "End, process "<< gc->rank();
	return 0;
}

