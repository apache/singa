#include <glog/logging.h>
#include <thread>
#include <memory>
#include <iostream>
#include "utils/singleton.h"
#include "utils/factory.h"
#include "trainer/worker.h"
#include "proto/model.pb.h"
using std::thread;
namespace singa {
Worker::Worker(int thread_id, int group_id, int worker_id):
  thread_id_(thread_id),group_id_(group_id), worker_id_(worker_id){
  }

void Worker::Setup(const ModelProto& model,
    shared_ptr<NeuralNet> train_net,
    shared_ptr<PMWorker::ParamShard> shard){
  train_net_=train_net;
  modelproto_=model;
  pmworker_=shared_ptr<PMWorker>(Singleton<Factory<PMWorker>>::Instance()
      ->Create("PMWorker"));
  pmworker_->Setup(group_id_, worker_id_, shard);
}

void Worker::Run(){
  param_dealer_=make_shared<Dealer>(2*thread_id_);
  param_dealer_->Connect(kInprocRouterEndpoint);
  param_poller_.Add(param_dealer_.get());
  layer_dealer_=make_shared<Dealer>(2*thread_id_+1);
  layer_dealer_->Connect(kInprocRouterEndpoint);

  {
  Msg* ping=new Msg();
  ping->set_src(group_id_, worker_id_, kWorkerParam);
  ping->set_dst(0,0,kStub);
  ping->set_type(kConnect);
  ping->add_frame("PING", 4);
  param_dealer_->Send(ping);
  ping=param_dealer_->Receive();
  string pong((char*)ping->frame_data(), ping->frame_size());
  CHECK_STREQ("PONG", pong.c_str());
  delete ping;
  }

  {
  Msg* ping=new Msg();
  ping->set_src(group_id_, worker_id_, kWorkerLayer);
  ping->set_dst(0,0,kStub);
  ping->set_type(kConnect);
  ping->add_frame("PING", 4);
  layer_dealer_->Send(ping);
  ping=layer_dealer_->Receive();
  string pong((char*)ping->frame_data(), ping->frame_size());
  CHECK_STREQ("PONG", pong.c_str());
  delete ping;
  }
  step_=modelproto_.step();
  // init params
  for(auto layer: train_net_->layers()){
    //LOG(ERROR)<<layer->partitionid()<<" : "<<layer->name();
    if(layer->partitionid()==worker_id_)
      for(auto param: layer->GetParams()){
        if(group_id_==0&&param->owner()==param->id()){
          param->Init(0);
          Put(param, step_);
        }else{
          Get(param, step_);
        }
      }
  }
  step_=modelproto_.step();
  Performance perf(train_net_);
  while(!StopNow(step_)){
    RunOneBatch(step_, &perf);
    step_++;
  }
}
int Worker::Put(shared_ptr<Param> param, int step){
  auto msg=pmworker_->Put(param, step);
  if(msg!=nullptr)
    param_dealer_->Send(msg);
  return 1;
}
int Worker::Get(shared_ptr<Param> param, int step){
  auto msg=pmworker_->Get(param, step);
  if(msg!=nullptr)
    param_dealer_->Send(msg);
  return 1;
}
int Worker::Update(shared_ptr<Param> param, int step){
  auto msg=pmworker_->Update(param, step);
  if(msg!=nullptr)
    param_dealer_->Send(msg);
  return 1;
}

int Worker::CollectAll(shared_ptr<NeuralNet> net, int step){
  auto& layers=net->layers();
  for(auto& layer: layers){
    if(layer->partitionid()==worker_id_)
      for(shared_ptr<Param> p: layer->GetParams()){
        Collect(p, step);
      }
  }
  return 1;
}
int Worker::Collect(shared_ptr<Param> param, int step){
  while(param->version()<step){
    Socket* which=param_poller_.Wait(10);
    if(which!=nullptr){
      Msg* msg=param_dealer_->Receive();
      if(msg==nullptr)
        return 0;
      pmworker_->Collect(&msg);
    }
  }
  return 1;
}

void Worker::RunOneBatch(int step, Performance* perf){
  //DLOG(ERROR)<<"Step "<<step;
  // Test will call Pull which updates the sync time
  // Hence we store the sync time, and restore it later
  //float tSyncData=tSyncData_, tSyncParam=tSyncParam_;
  if(ValidateNow(step)){
    LOG(ERROR)<<"Validation at step "<<step;
    CollectAll(validation_net_, step);
    Test(validation_net_, modelproto_.validation_steps(), perf!=nullptr);
  }
  if(TestNow(step)){
    LOG(ERROR)<<"Test at step "<<step;
    CollectAll(test_net_, step);
    Test(test_net_, modelproto_.test_steps(), perf!=nullptr);
  }
  //tSyncData_=tSyncData; tSyncParam_=tSyncParam;

  CollectAll(train_net_, step);
  TrainOneBatch(step);
  if(perf!=nullptr){
    perf->Update();
    if(DisplayNow(step)){
      LOG(ERROR)<<"Training at step "<<step;
      LOG(ERROR)<<"\t"<<perf->ToString();
      perf->Reset();
      //LOG(ERROR)<<"\t"<<TimerInfo();
    }
  }

  /*
  if(CheckpointNow(step)){
    pm_->Checkpoint(cluster_->workspace()+"/snapshot-"+std::to_string(step));
  }
  */
}

void Worker::ReceiveBlobs(shared_ptr<NeuralNet> net){
  /*
  int type;
  char *name;
  int64_t tick=zclock_mono();
  zframe_t* frame=zframe_new_empty();

  zsock_recv(pull_, "isf", &type, &name, &frame);
  if(type==kDataFrame){
    auto* dst=static_cast<BridgeDstLayer*>(
        net->name2layer(string(name)).get());
    memcpy(dst->mutable_data()->mutable_cpu_data(), zframe_data(frame),
        zframe_size(frame));
    dst->set_ready(true);
  }else if(type==kGradFrame){
    auto* src=static_cast<BridgeSrcLayer*>(net->name2layer(string(name)).get());
    memcpy(src->mutable_grad()->mutable_cpu_data(), zframe_data(frame),
        zframe_size(frame));
    src->set_ready(true);
  }
  zframe_destroy(&frame);
  delete name;
  tSyncData_+=zclock_mono()-tick;
  */
}

void Worker::SendBlob(){

}

void Worker::Test(shared_ptr<NeuralNet> net, int nsteps, bool disperf){
  Performance perf(net);
  for(int step=0;step<nsteps;step++){
    TestOneBatch(net, step, kTest);
    if(disperf)
      perf.Update();
  }
  if(disperf)
    LOG(ERROR)<<"\t"<<perf.ToString();
}

/****************************BPWorker**********************************/

void BPWorker::Forward(shared_ptr<NeuralNet> net, int step,  bool training){
  auto& layers=net->layers();
  for(auto& layer: layers){
    if(layer->partitionid()==worker_id_){
      if(layer->is_bridgedstlayer()){
        auto* dst=static_cast<BridgeDstLayer*>(layer.get());
        while(!dst->ready()){
          auto msg=layer_dealer_->Receive();
          CHECK_EQ(msg->src_group_id(), group_id_);
          string name((char*)msg->frame_data(), msg->frame_size());
          auto tmp=net->name2layer(name);
          CHECK(tmp->is_bridgedstlayer());
          auto* dstlayer=static_cast<BridgeDstLayer*>(tmp.get());
          auto data=dstlayer->mutable_data(nullptr);
          msg->next_frame();
          memcpy(data->mutable_cpu_data(), msg->frame_data(), msg->frame_size());
          dstlayer->set_ready(true);
          delete msg;
        }
      }
      if(training){
        for(shared_ptr<Param> p: layer->GetParams()){
          Collect(p, step);
        }
      }
      //clock_t s=clock();
      layer->ComputeFeature(training);
      //LOG(ERROR)<<layer->name()<<":"<<(clock()-s)*1.0/CLOCKS_PER_SEC;
      if(layer->is_bridgesrclayer()){
        auto dst=layer->dstlayers().at(0);
        Msg *msg=new Msg();
        msg->set_src(group_id_, worker_id_, kWorkerLayer);
        msg->set_dst(group_id_, dst->partitionid(), kWorkerLayer);
        msg->add_frame(dst->name().c_str(), dst->name().length());
        auto const & blob=layer->data(nullptr);
        msg->add_frame(blob.cpu_data(), blob.count()*sizeof(float));
        layer_dealer_->Send(msg);
      }
      if(training&&DisplayDebugInfo(step)&&layer->mutable_data(nullptr)!=nullptr){
        LOG(INFO)<<StringPrintf("Forward layer  %10s data norm1 %13.9f",
            layer->name().c_str(), layer->data(nullptr).asum_data());
      }
    }
  }
}

void BPWorker::Backward(shared_ptr<NeuralNet> net, int step){
  auto& layers=net->layers();
  for (auto it = layers.rbegin(); it != layers.rend(); it++){
    shared_ptr<Layer> layer=*it;
    if(layer->partitionid()==worker_id_){
      if(layer->is_bridgesrclayer()){
        //auto* src=static_cast<BridgeSrcLayer*>(layer.get());
        // receive grad blobs
      }
      layer->ComputeGradient();
      if(DisplayDebugInfo(step)&&layer->mutable_grad(nullptr)!=nullptr){
        LOG(INFO)<<StringPrintf("Backward layer %10s grad norm1 %13.9f\t",
            layer->name().c_str(), layer->grad(nullptr).asum_data());
        for(shared_ptr<Param> p: layer->GetParams())
          LOG(INFO)<<StringPrintf("param id %2d, name %10s,\
              value norm1 %13.9f, grad norm1 %13.9f",
              p->id(), p->name().c_str(),
              p->data().asum_data(), p->grad().asum_data());
      }
      for(shared_ptr<Param> p: layer->GetParams()){
        Update(p, step);
      }
      if(layer->is_bridgedstlayer()){
        // send grad blobs
      }
    }
  }
}

void BPWorker::TrainOneBatch(int step){
  Forward(train_net_, step, true);
  Backward(train_net_, step);
}

void BPWorker::TestOneBatch(shared_ptr<NeuralNet> net,int step, Phase phase){
  Forward(net, step, false);
}

/*********************Implementation for Performance class*******************/
Performance::Performance(shared_ptr<NeuralNet> net):net_(net), counter_(0){
  for(auto& layer: net->losslayers()){
    name_.push_back(layer->name());
    metric_.push_back(vector<float>{});
    metric_.back().resize(layer->metric().count(),0.f);
  }
}

void Performance::Update(){
  const auto& losslayers=net_->losslayers();
  for(size_t i=0;i<losslayers.size();i++){
    const float * ptr=losslayers[i]->metric().cpu_data();
    vector<float>& m=metric_.at(i);
    for(int j=0;j<losslayers[i]->metric().count();j++)
      m[j]+=ptr[j];
  }
  counter_++;
}

void Performance::Reset(){
  for(auto& m: metric_)
    for(auto& x: m)
      x=0.f;
  counter_=0;
}

string Performance::ToString(){
  string disp="";
  for(size_t i=0;i<metric_.size();i++){
    disp+="Output from "+name_[i]+" layer ";
    vector<float> m=metric_.at(i);
    for(size_t j=0;j<m.size();j++)
        disp+=std::to_string(j)+" : "+std::to_string(m[j]/counter_)+"\t";
    disp+="\n";
  }
  return disp;
}
/*
void Executor::Setup(int local_threadid, const ModelProto& model){
  tForward_=tBackward_=tSyncData_=tSyncParam_=0;
  modelproto_=model;
  local_threadid_=local_threadid;
  if(model.prefetch()){
    for(auto& layer: train_net_->datalayers()){
      if(cluster_->group_threadid(local_threadid_)==layer->locationid())
        localDataLayers_.push_back(layer);
    }
    if(localDataLayers_.size())
      prefetch_thread_=std::thread(Executor::PrefetchData,
          std::ref(localDataLayers_), true,1);
  }
  int gthreadid=cluster_->group_threadid(local_threadid);
}

void Executor::PrefetchData(const vector<DataLayer*>& datalayers, bool training,
    int steps){
  if(datalayers.size()==0)
    return;
  for(int i=0;i<steps;i++){
    for(auto& layer: datalayers){
      layer->Prefetching(training);
      for(auto& dstlayer: layer->dstlayers()){
        CHECK(dstlayer->is_parserlayer());
        auto parserlayer=static_cast<ParserLayer*>(dstlayer.get());
        parserlayer->Prefetching(training);
      }
    }
  }
}
*/

}  // namespace singa
