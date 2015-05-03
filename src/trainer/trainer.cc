#include <thread>
#include <vector>
#include <map>
#include <glog/logging.h>
#include "trainer/trainer.h"
using std::vector;
using std::map;

namespace singa {
int ProcsIDOf(int group_id, int id, int flag){
  int procsid;
  auto cluster=Cluster::Get();
  if(flag==kServer){
    procsid=group_id*cluster->nservers_per_group()/
      cluster->nservers_per_procs()+id/cluster->nservers_per_procs();
    if(cluster->server_worker_separate())
      procsid+=cluster->nworker_procs();
  }else if(flag==kWorkerLayer || flag==kWorkerParam){
    procsid=group_id*cluster->nworkers_per_group()
      /cluster->nworkers_per_procs();
    if(cluster->nworkers_per_group()>cluster->nworkers_per_procs())
      procsid+=id/cluster->nworkers_per_procs();
  }else{
    LOG(ERROR)<<"Unkown flag ("<<flag<<")";
  }
  return procsid;
}

void Trainer::RegisterDefaultClasses(const singa::ModelProto& proto){
  // register all layers appearing in the neural net
  singa::NeuralNet::RegisterLayers();
  Singleton<Factory<singa::Param>>::Instance()->Register(
      "Param", CreateInstance(singa::Param, singa::Param));
  Singleton<Factory<singa::Updater>>::Instance() ->Register(
      "Updater", CreateInstance(singa::SGDUpdater, singa::Updater));
  Singleton<Factory<singa::PMWorker>>::Instance() ->Register(
      "PMWorker", CreateInstance(singa::PMWorker, singa::PMWorker));
  Singleton<Factory<singa::PMServer>>::Instance() ->Register(
      "PMServer", CreateInstance(singa::PMServer, singa::PMServer));
  Singleton<Factory<singa::PMServer>>::Instance() ->Register(
      "PMServer", CreateInstance(singa::PMServer, singa::PMServer));
}

void Trainer::Start(const ModelProto& mproto, const ClusterProto& cproto,
    int procs_id){
  RegisterDefaultClasses(mproto);

  auto cluster=Cluster::Get(cproto, procs_id);
  // create servers
  vector<shared_ptr<Server>> servers;
  int nSocket=1; // the first socket is the router
  if(cluster->has_server()){
    int pid=cluster->procs_id();
    if(cluster->server_worker_separate())
      pid-=cluster->nworker_procs();
    int gid=pid*cluster->nservers_per_procs()/cluster->nservers_per_group();
    int start=pid*cluster->nservers_per_procs()%cluster->nservers_per_group();
    int end=start+cluster->nservers_per_group();
    // the ParamShard for servers consists of a dictionary of Param objects
    auto shard=make_shared<PMServer::ParamShard>();
    for(int sid=start;sid<end;sid++){
      auto server=make_shared<Server>(gid, sid);
      auto dealer=make_shared<Dealer>(nSocket++);
      dealer->Connect(kInprocRouterEndpoint);
      server->Setup(mproto.updater(), shard, dealer);
      servers.push_back(server);
    }
  }

  // create workers
  vector<shared_ptr<Worker>> workers;
  if(cluster->has_worker()){
    auto net=NeuralNet::SetupNeuralNet(mproto.neuralnet(), kTrain);
    int pid=cluster->procs_id();
    int gstart, gend, wstart, wend;
    if(cluster->nworkers_per_group()>=cluster->nworkers_per_procs()){
      // all workers in this procs are from the same group
      gstart=pid*cluster->nworkers_per_procs()/cluster->nworkers_per_group();
      gend=gstart+1;
      wstart=pid*cluster->nworkers_per_procs()%cluster->nworkers_per_group();
      wend=wstart+cluster->nworkers_per_group();
    }else{
      // there are multiple groups in this procs
      CHECK_EQ(cluster->nworkers_per_procs()%cluster->nworkers_per_group(),0);
      int groups_per_procs=
        cluster->nworkers_per_procs()/cluster->nworkers_per_group();
      gstart=pid*groups_per_procs;
      gend=(pid+1)*groups_per_procs;
      wstart=0;
      wend=cluster->nworkers_per_group();
    }
    for(int gid=gstart;gid<gend;gid++){
      shared_ptr<NeuralNet> train_net, test_net, validation_net;
      if(gid==gstart)
        train_net=net;
      else{
        train_net=NeuralNet::SetupNeuralNet(mproto.neuralnet(), kTrain);
        // the train net for other groups may share parameter values from the
        // first group
        if(mproto.hogwild())
          train_net->ShareParams(net, kValueOnly);
      }
      if(gid==0){
        // validation and test are performed only by the first group
        if(mproto.test_steps()){
          test_net=NeuralNet::SetupNeuralNet(mproto.neuralnet(), kTest);
          if(test_net!=nullptr)
            test_net->ShareParams(train_net, kValueOnly);
        }
        if(mproto.validation_steps()){
          validation_net=NeuralNet::SetupNeuralNet(mproto.neuralnet(), kValidation);
          if(validation_net!=nullptr)
            validation_net->ShareParams(train_net, kValueOnly);
        }
      }
      // create ParamShard for the workers
      auto shard=make_shared<PMWorker::ParamShard>();
      for(auto layer: train_net->layers()){
        int procsid=ProcsIDOf(gid, layer->locationid(),kWorkerParam);
        int local=procsid==cluster->procs_id();
        for(auto param: layer->GetParams()){
          int owner=param->owner()<0||param->owner()==param->id()?procsid:-1;
          if(shard->find(param->id())==shard->end())
            (*shard)[param->id()]=make_shared<ParamCounter>(param, local, owner);
          else
            shard->at(param->id())->AddParam(param, local, owner);
        }
      }
      for(int wid=wstart;wid<wend;wid++){
        shared_ptr<Worker> worker=nullptr;
        if(mproto.alg()==ModelProto_GradCalcAlg_kBackPropagation)
          worker=make_shared<BPWorker>(gid, wid);
        else{
        // TODO add CDWorker
        }
        auto layer_dealer=make_shared<Dealer>(nSocket++);
        auto param_dealer=make_shared<Dealer>(nSocket++);
        layer_dealer->Connect(kInprocRouterEndpoint);
        param_dealer->Connect(kInprocRouterEndpoint);
        worker->Setup(mproto, train_net, shard, layer_dealer, param_dealer);
        worker->set_test_net(test_net);
        worker->set_validation_net(validation_net);
        workers.push_back(worker);
      }
    }
  }

#ifdef USE_MPI
  for(int i=0;i<nSocket;i++){
    MPIQueues.push_back(make_shared<SafeQueue>());
  }
#endif
  vector<std::thread> threads;
  for(auto server: servers)
    threads.push_back(std::thread(&Server::Run,server));
  for(auto worker: workers)
    threads.push_back(std::thread(&Worker::Run,worker));
  Run();
  for(auto& thread: threads)
    thread.join();
}

void Trainer::Run(){
  auto cluster=Cluster::Get();
  auto router=make_shared<Router>();
  router->Bind(kInprocRouterEndpoint);
  if(cluster->nprocs()>1)
    router->Bind(cluster->endpoint());

  map<int, shared_ptr<Dealer>> interprocs_dealers;
  Poller poller;
  poller.Add(router.get());
  int timeout=cluster->stub_timeout();
  while(true){
    Msg* msg=router->Receive();
    if(msg==nullptr){
      LOG(ERROR)<<"Connection broken!";
      exit(0);
    }
    int dst_flag=msg->dst_flag();
    int type=msg->type();
    int group_id, id, procs_id;
    switch (dst_flag){ // TODO process other requests, e.g. RESTful
      case kStub:
        if(type==kConnect){
          delete msg;
        }else{
          // TODO processing requests for worker group spanning multiple procs.
          LOG(ERROR)<<"Unkown message type ("<<type<<") to stub";
        }
        break;
      default:
        group_id=msg->dst_group_id();
        id=msg->dst_id();
        procs_id=ProcsIDOf(group_id, id, dst_flag);
        if(procs_id!=cluster->procs_id()){
          if (interprocs_dealers.find(procs_id)==interprocs_dealers.end())
            interprocs_dealers[procs_id]=make_shared<Dealer>(procs_id);
          interprocs_dealers[procs_id]->Send(msg);
        } else
          router->Send(msg);
        break;
    }
  }
}
} /* singa */
