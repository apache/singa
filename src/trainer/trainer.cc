#include <thread>
#include <vector>
#include <map>
#include <queue>
#include <chrono>
#include <glog/logging.h>
#include "proto/common.pb.h"
#include "trainer/trainer.h"
#include "mshadow/tensor.h"
using std::vector;
using std::map;
using namespace std::chrono;

typedef std::chrono::milliseconds TimeT;

namespace singa {

void Trainer::RegisterDefaultClasses(const singa::ModelProto& proto){
  // register all layers appearing in the neural net
  singa::NeuralNet::RegisterLayers();
  Singleton<Factory<singa::Param>>::Instance()->Register(
      "Param", CreateInstance(singa::Param, singa::Param));
  Singleton<Factory<singa::Updater>>::Instance() ->Register(
      "Updater", CreateInstance(singa::SGDUpdater, singa::Updater));
}

void HandleWorkerFinish(void * ctx){
  HandleContext* hctx=static_cast<HandleContext*> (ctx);
  Msg* msg=new Msg();
  msg->set_src(-1,-1, kRuntime);
  msg->set_dst(hctx->group_id, hctx->id, kServer);
  msg->set_type(kStop);
  hctx->dealer->Send(&msg);
}

const std::unordered_map<int, vector<std::pair<int, int>>> SliceParams(int num,
    const vector<shared_ptr<Param>>& params){
  std::unordered_map<int, vector<std::pair<int, int>>> paramid2slices;
  if (num==0)
    return paramid2slices;
  vector<int> param_size;
  int avg=0;
  for(const auto& x:params){
    if(x->owner()==x->id())
      avg+=x->size();
  }
  avg=avg/num+avg%num;
  int diff=avg/10;
  LOG(INFO)<<"Slicer, param avg="<<avg<<", diff= "<<diff;

  int capacity=avg, sliceid=0, nbox=0;
  for(auto& param: params){
    if(param->id()!=param->owner())
      continue;
    int x=param->size(), paramid=param->id();
    LOG(INFO)<<"param id="<<paramid<<", total size="<<x;
    while(x>0){
      int size=0;
      if(capacity>=x){
        capacity-=x;
        size=x;
        x=0;
      }else if(capacity+diff>=x){
        size=x;
        x=0;
        capacity=0;
      }else if(capacity>=diff){
        x-=capacity;
        size=capacity;
        capacity=avg;
        nbox++;
      }else{
        capacity=avg;
        nbox++;
      }
      if(size){
        paramid2slices[paramid].push_back(std::make_pair(sliceid++, size));
        LOG(INFO)<<"param id="<<paramid<<", slice size="<<size;
      }
    }
  }
  CHECK_LE(nbox, num);
  return paramid2slices;
}
const vector<int> PartitionSlice(int num, const vector<int>& slices){
  int avg=0;
  for(int x: slices)
    avg+=x;
  avg=avg/num+avg%num;
  int box=avg, boxid=0, diff=avg/10;
  vector<int> slice2box;
  for(auto it=slices.begin(); it!=slices.end();){
    int x=*it;
    if(box>=x){
      box-=x;
      slice2box.push_back(boxid);
      it++;
    }else if(box+diff>=x){
      slice2box.push_back(boxid);
      it++;
      box=0;
    }else{
      box=avg;
      boxid++;
    }
  }
//  CHECK_LT(slice2box.back(), num);
  CHECK_EQ(slice2box.size(), slices.size());
  int previd=slice2box[0];
  std::string disp;
  for(size_t i=0;i<slice2box.size();i++)
    if(previd!=slice2box[i]){
      disp+=", "+std::to_string(slices[i]);
      previd=slice2box[i];
    } else
      disp+=" "+std::to_string(slices[i]);
  LOG(INFO)<<"partition slice (av ="<<avg<<", num="<<num<<"):"<<disp;
  return slice2box;
}
vector<shared_ptr<Server>> Trainer::CreateServers(int nthreads,
    const ModelProto & mproto,
    const vector<int> slices,
    vector<HandleContext*>* ctx){
  auto cluster=Cluster::Get();
  vector<shared_ptr<Server>> servers;
  if(!cluster->has_server())
    return servers;

  int pid=cluster->procs_id();
  if(cluster->server_worker_separate())
    pid-=cluster->nworker_procs();
  int gid=pid*cluster->nservers_per_procs()/cluster->nservers_per_group();
  int start=pid*cluster->nservers_per_procs()%cluster->nservers_per_group();
  int end=start+cluster->nservers_per_procs();
  // the ServerShard for servers consists of a dictionary of Param objects
  server_shard_=make_shared<ServerShard>();
  auto slice2group=PartitionSlice(cluster->nserver_groups(), slices);
  if(start<end){
    auto dealer=make_shared<Dealer>();
    dealer->Connect(kInprocRouterEndpoint);
    for(int sid=start;sid<end;sid++){
      auto server=make_shared<Server>(nthreads++, gid, sid);
      server->Setup(mproto.updater(), server_shard_, slice2group);
      servers.push_back(server);
      auto *hc=new HandleContext{dealer, gid, sid};
      ctx->push_back(hc);
      CHECK(cluster->runtime()->WatchSGroup(gid, sid, HandleWorkerFinish,
            ctx->back()));
    }
  }
  return servers;
}

vector<shared_ptr<Worker>> Trainer::CreateWorkers(int nthreads,
    const ModelProto& mproto, vector<int> *slice_size){
  auto cluster=Cluster::Get();
  auto net=NeuralNet::SetupNeuralNet(mproto.neuralnet(), kTrain,
      cluster->nworkers_per_group());
  int lcm=LeastCommonMultiple(cluster->nserver_groups(), cluster->nservers_per_group());
  auto paramid2slices=SliceParams(lcm, net->params()); // sliceid, size
  for(auto param: net->params()){
    if(param->id()==param->owner())
      for(auto entry: paramid2slices[param->id()])
        slice_size->push_back(entry.second);
  }

  vector<shared_ptr<Worker>> workers;
  if(!cluster->has_worker())
    return workers;
  //LOG(ERROR)<<net->ToString();
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
      train_net=NeuralNet::SetupNeuralNet(mproto.neuralnet(), kTrain,
          cluster->nworkers_per_group());
      // the train net for other groups may share parameter values from the
      // first group
      if(cluster->share_memory())
        train_net->ShareParams(net, kValueOnly);
    }
    if(gid==0){
      // validation and test are performed only by the first group
      if(mproto.test_steps()){
        test_net=NeuralNet::SetupNeuralNet(mproto.neuralnet(), kTest,
            cluster->nworkers_per_group());
        if(test_net!=nullptr)
          test_net->ShareParams(train_net, kValueOnly);
      }
      if(mproto.validation_steps()){
        validation_net=NeuralNet::SetupNeuralNet(mproto.neuralnet(), kValidation,
            cluster->nworkers_per_group());
        if(validation_net!=nullptr)
          validation_net->ShareParams(train_net, kValueOnly);
      }
    }
    // create ServerShard for the workers
    auto shard=make_shared<WorkerShard>();
    worker_shards_[gid]=shard;
    for(auto layer: train_net->layers()){
      int procsid=cluster->ProcsIDOf(gid, layer->partitionid(), kWorkerLayer);
      bool local=procsid==cluster->procs_id();
      for(auto param: layer->GetParams()){
        for(auto entry :paramid2slices[param->owner()]){
          param->AddSlice(entry.first,  entry.second);
        }
        int owner_procs=param->owner()==param->id()?procsid:procs_id_;
        if(shard->find(param->owner())==shard->end())
          (*shard)[param->owner()]=
            make_shared<ParamInfo>(param, local, owner_procs);
        else
          shard->at(param->owner())->AddParam(param, local);
      }
    }
    for(int wid=wstart;wid<wend;wid++){
      shared_ptr<Worker> worker=nullptr;
      if(mproto.alg()==ModelProto_GradCalcAlg_kBackPropagation)
        worker=make_shared<BPWorker>(nthreads++,gid, wid);
      else{
        worker=make_shared<CDWorker>(nthreads++,gid, wid);
      }
      worker->Setup(mproto, train_net);
      worker->set_test_net(test_net);
      worker->set_validation_net(validation_net);
      workers.push_back(worker);
    }
  }
  return workers;
}

void Trainer::Start(const ModelProto& mproto, const ClusterProto& cproto,
    int procs_id){
  // procs_id is only used for resume training
  CHECK_EQ(procs_id, -1);
  RegisterDefaultClasses(mproto);

  auto cluster=Cluster::Get(cproto, procs_id);
  router_=make_shared<Router>();
  router_->Bind(kInprocRouterEndpoint);
  if(cluster->nprocs()>1){
    const string hostip=cluster->hostip();
    int port=router_->Bind("tcp://"+hostip+":*");
    cluster->Register(hostip+":"+std::to_string(port));
  }else
    cluster->set_procs_id(0);

  procs_id_ = cluster->procs_id();
  int nthreads=1;
  // create workers
  vector<int> slices;
  vector<shared_ptr<Worker>> workers=CreateWorkers(nthreads, mproto, &slices);
  if(cluster->nserver_groups()&&cluster->nservers_per_group())
    slice2server_=PartitionSlice(cluster->nservers_per_group(), slices);
  nthreads+=workers.size();
  // create servers
  vector<HandleContext*> ctx;
  vector<shared_ptr<Server>> servers=CreateServers(nthreads, mproto, slices,
      &ctx);

#ifdef USE_MPI
  for(int i=0;i<nSocket;i++){
    MPIQueues.push_back(make_shared<SafeQueue>());
  }
#endif
  vector<std::thread> threads;
  for(auto server: servers)
    threads.push_back(std::thread(&Server::Run,server.get()));
  for(auto worker: workers)
    threads.push_back(std::thread(&Worker::Run,worker.get()));
  Run(workers, servers);
  for(auto& thread: threads)
    thread.join();
  for(auto x: ctx)
    delete x;
}

inline int bandwidth(int bytes, system_clock::time_point start){
  auto now=system_clock::now();
  auto duration=duration_cast<TimeT> (now - start);
  return static_cast<int>(bytes*1000.f/duration.count());
}

void Trainer::Run(const vector<shared_ptr<Worker>>& workers,
    const vector<shared_ptr<Server>>& servers){
  auto cluster=Cluster::Get();
  procs_id_=cluster->procs_id();
  LOG(INFO)<<"Stub in process "<<procs_id_<<" starts";
  map<int, shared_ptr<Dealer>> interprocs_dealers;
  std::queue<Msg*> msg_queue;
  bool stop=false;
  auto start=std::chrono::system_clock::now();
  float amount=0.f;
  Poller poll;
  poll.Add(router_.get());
  int sync_server=0, nworkers=workers.size(), nservers=servers.size();
  while(!stop){
    // if the poll time is large, then the poller may not expire
    // if it is small, then many reminder messages will be sent which may
    // slow done the process of other request. TODO tune it.
    auto *sock=poll.Wait(cluster->poll_time());
    if(poll.Terminated()){
      LOG(ERROR)<<"Connection broken!";
      exit(0);
    }else if(sock==nullptr){
      if(cluster->nserver_groups()>1&&
          bandwidth(amount, start)<cluster->bandwidth()){
        Msg* msg=new Msg();
        msg->set_src(-1,-1, kStub);
        msg->set_dst(servers[sync_server]->group_id(),
            servers[sync_server]->server_id(), kServer);
        msg->set_type(kSyncReminder);
        sync_server=(sync_server+1)%servers.size();
        router_->Send(&msg);
      }
      continue;
    }
    Msg* msg=router_->Receive();
    if(msg==nullptr){
      LOG(ERROR)<<"Connection broken!";
      exit(0);
    }
    msg_queue.push(msg);
    while(!msg_queue.empty()){
      msg=msg_queue.front();
      msg_queue.pop();
      int dst_flag=msg->dst_flag();
      int type=msg->type();
      int dst_procs=msg->dst_first();
      if(dst_flag == kStub&&(dst_procs==procs_id_||dst_procs==-1)){
        if(type==kConnect){
          msg_queue.push(HandleConnect(&msg));
        }else if(type==kStop){
          if(msg->src_flag()==kServer)
            nservers--;
          else if (msg->src_flag()==kWorkerParam)
            nworkers--;
          DeleteMsg(&msg);
          if(nworkers==0&&nservers==0){
            stop=true;
            break;
          }
        }else if(type==kMetric){
          if(msg->src_first()==0){
            int step=msg->trgt_first();
            string prefix((char*)msg->frame_data(), msg->frame_size());
            msg->next_frame();
            Metric cur;
            cur.ParseString(string((char*)msg->frame_data(), msg->frame_size()));
            LOG(ERROR)<<prefix<<" step-" <<step<<", "<<cur.ToString();
          }
          DeleteMsg(&msg);
        }else if(cluster->nserver_groups()>0){
          int group_id;
          int paramid=msg->trgt_first();
          shared_ptr<ParamInfo> entry;
          switch (type){ // TODO process other requests, e.g. RESTful
            case kUpdate:
              group_id=msg->src_first();
              entry=worker_shards_.at(group_id)->at(paramid);
              for(auto x:HandleUpdate(entry, &msg))
                msg_queue.push(x);
              break;
            case kRUpdate:
              group_id=msg->dst_second();
              entry=worker_shards_.at(group_id)->at(paramid);
              HandleUpdateResponse(entry, &msg);
              break;
            case kGet:
              group_id=msg->src_first();
              entry=worker_shards_.at(group_id)->at(paramid);
              for(auto x:HandleGet(entry, &msg))
                msg_queue.push(x);
              break;
            case kRGet:
              group_id=msg->dst_second();
              entry=worker_shards_.at(group_id)->at(paramid);
              HandleGetResponse(entry, &msg);
              break;
            case kPut:
              group_id=msg->src_first();
              entry=worker_shards_.at(group_id)->at(paramid);
              for(auto x:HandlePut(entry, &msg))
                msg_queue.push(x);
              break;
            default:
              LOG(ERROR)<<"Unknow message type:"<<type;
              break;
          }
        }else{
          DeleteMsg(&msg);
        }
      }else{
        int dst_procs_id;
        if(dst_flag==kStub){
          dst_procs_id=msg->dst_first();
        }else{
          dst_procs_id=cluster->ProcsIDOf(msg->dst_first(),
              msg->dst_second(), msg->dst_flag());
        }
        if(dst_procs_id!=procs_id_){
          // forward to other procs
          if (interprocs_dealers.find(dst_procs_id)==interprocs_dealers.end()){
            auto dealer=make_shared<Dealer>();
            interprocs_dealers[dst_procs_id]=dealer;
            while(cluster->endpoint(dst_procs_id)==""){
              std::this_thread::sleep_for(
                  std::chrono::milliseconds(3000));//kCollectSleepTime));
              LOG(ERROR)<<"waiting for procs "<< dst_procs_id<<" to register";
            }
            dealer->Connect("tcp://"+cluster->endpoint(dst_procs_id));
          }
          if(bandwidth(amount, start) <=cluster->bandwidth()){
            start=std::chrono::system_clock::now();
            amount=0;
          }
          amount+=msg->size();
          //LOG(ERROR)<<"send inter msg of type "<<msg->type();
          interprocs_dealers[dst_procs_id]->Send(&msg);
        }else{
          if(type==kSyncRequest){
            char buf[32];
            sprintf(buf, "%d", cluster->bandwidth()-bandwidth(amount, start));
            msg->add_frame(buf, strlen(buf));
          }
          router_->Send(&msg);
        }
      }
    }
  }
  LOG(INFO)<<"Stub in process "<<procs_id_<<" stops";
}
Msg* Trainer::HandleConnect(Msg** msg){
  string ping((char*)(*msg)->frame_data(), (*msg)->frame_size());
  CHECK_STREQ("PING", ping.c_str());
  // ping-pong for debug
  (*msg)->SwapAddr();
  Msg* reply=new Msg();
  reply->SetAddr(*msg);
  reply->add_frame("PONG", 4);
  reply->set_type(kConnect);
  DeleteMsg(msg);
  return reply;
}
const vector<Msg*> Trainer::HandleGet(shared_ptr<ParamInfo> pi, Msg** msg){
  Msg* msgg=*msg;
  vector<Msg*> replies;
  int version=msgg->trgt_third();
  if(msgg->src_flag()==kStub){
    LOG(FATAL)<<"Not implemented";
    /*
    if(version<=pi->shares.at(0)->version()){
      replies.push_back(pi->shares.at(0)->HandleGetMsg(msg));
    }else if(version>pi->next_version){
      // reinsert into a msg queue.
      replies.push_back(mmsg);
    }
    */
  }else if(version>pi->next_version){
    pi->next_version=version;
    int gid=msgg->src_first();
    int group=gid/Cluster::Get()->nworker_groups_per_server_group();
    auto param=pi->shares.at(0);
    for(int idx=0, id=param->slice_start();idx<param->num_slices();idx++){
      int server=slice2server_[id+idx];
      int procs=Cluster::Get()->ProcsIDOf(group, server, kServer);
      auto x=param->GenGetMsg(procs!=procs_id_, idx);
      x->set_trgt(param->owner(), id+idx, param->local_version()+1);
      x->set_src(procs_id_, gid, kStub);
      x->set_dst(group, server, kServer);
      //LOG(ERROR)<<"stub handle get for "<<idx+id<<","<<group<<","<<server;
      replies.push_back(x);
    }
  }
  return replies;
}

const vector<Msg*> Trainer::HandleUpdate(shared_ptr<ParamInfo>pi, Msg** msg){
  Msg* msgg=*msg ;
  vector<Msg*> ret;
  int step= msgg->trgt_third();
  if(msgg->src_flag()==kStub){
    if(pi->num_update<pi->num_local){
      ret.push_back(*msg);
      return ret; //wait unitl local updates are ready
    }
    int n; sscanf((char*)(*msg)->frame_data(), "%d", &n);
    pi->num_update+=n;
    auto it=pi->shares.begin();
    auto shape=mshadow::Shape1((*it)->size());
    mshadow::Tensor<mshadow::cpu,1> agg((*it)->mutable_cpu_grad(), shape);
    mshadow::Tensor<mshadow::cpu,1> grad((*it)->mutable_cpu_grad(), shape);
    agg+=grad;
  }else if(++pi->num_update>=pi->num_local){
    auto it=pi->shares.begin();
    auto shape=mshadow::Shape1((*it)->size());
    mshadow::Tensor<mshadow::cpu,1> agg((*it)->mutable_cpu_grad(), shape);
    for(++it;it!=pi->shares.end();it++){
      mshadow::Tensor<mshadow::cpu,1> grad((*it)->mutable_cpu_grad(), shape);
      agg+=grad;
    }
    agg/=pi->num_total;
    if(pi->num_local<pi->num_total){
      /*
      int gid=msgg->src_first();
      for(auto update: pi->shares.at(0)->GenUpdateMsg(step)){
        update->set_src(procs_id_, gid,kStub);
        update->set_dst(pi->owner_procs, gid, kStub);
        ret.push_back(update);
      }
      pi->num_update=0;
      */
    }
  }
  if(pi->num_update==pi->num_total){
    auto param=pi->shares.at(0);
    int group=msgg->src_first()/Cluster::Get()->nworker_groups_per_server_group();
    int srcgid=msgg->src_first();
    for(int idx=0, id=param->slice_start(); idx<param->num_slices();idx++){
      int server=slice2server_[idx+id];
      int procs=Cluster::Get()->ProcsIDOf(group, server, kServer);
      auto x=param->GenUpdateMsg(procs!=procs_id_, idx);
      x->set_trgt(param->owner(), id+idx, step);
      x->set_src(procs_id_, srcgid, kStub);
      x->set_dst(group, server, kServer);
      ret.push_back(x);
    }
    pi->num_update=0;
  }
  DeleteMsg(msg);
  return ret;
}

const vector<Msg*> Trainer::HandlePut(shared_ptr<ParamInfo>pi, Msg** msg){
  vector<Msg*> ret;
  CHECK_NE((*msg)->src_flag(), kStub);
  int gid=(*msg)->src_first();
  int version=(*msg)->trgt_third();
  auto param=pi->shares.at(0);
  int group=gid/Cluster::Get()->nworker_groups_per_server_group();
  for(int idx=0, start=param->slice_start();idx<param->num_slices(); idx++){
    int server=slice2server_[start+idx];
    int procs=Cluster::Get()->ProcsIDOf(group, server, kServer);
    auto x=param->GenPutMsg(procs!=procs_id_, idx);
    x->set_trgt(param->owner(), start+idx, version);
    x->set_src(procs_id_, gid, kStub);
    x->set_dst(group, server, kServer);
    ret.push_back(x);
    //LOG(ERROR)<<"stub handle put "<<start+idx<<"to "<<group<<","<<server;
  }
  DeleteMsg(msg);
  return ret;
}

void Trainer::HandleGetResponse(shared_ptr<ParamInfo>pi, Msg** msg){
  int version=(*msg)->trgt_third();
  int sliceid=(*msg)->trgt_second();
  auto param=pi->shares.at(0);
  if(param->ParseGetResponseMsg(msg,sliceid-param->slice_start()))
    param->set_version(version);
  // process get requests in waiting queue
}


void Trainer::HandleUpdateResponse(shared_ptr<ParamInfo> pi, Msg** msg){
  int sliceid=(*msg)->trgt_second();
  int version=(*msg)->trgt_third();
  auto param=pi->shares.at(0);
  if(param->ParseUpdateResponseMsg(msg,sliceid-param->slice_start())){
    param->set_version(version);
  }
}
} /* singa */
