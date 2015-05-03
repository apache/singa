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
Worker::Worker( int group_id, int worker_id):
   group_id_(group_id), worker_id_(worker_id){
}

void Worker::Setup(const ModelProto& model,
    shared_ptr<NeuralNet> train_net,
    shared_ptr<PMWorker::ParamShard> shard,
    shared_ptr<Dealer> layer_dealer,
    shared_ptr<Dealer> param_dealer){
  train_net_=train_net;
  modelproto_=model;
  layer_dealer_=layer_dealer;
  param_dealer_=param_dealer;
  if(layer_dealer_!=nullptr)
    layer_poller_.Add(layer_dealer_.get());
  if(param_dealer_!=nullptr)
    param_poller_.Add(param_dealer_.get());
  pmworker_=shared_ptr<PMWorker>(Singleton<Factory<PMWorker>>::Instance()
      ->Create("PMWorker"));
  pmworker_->Setup(group_id_, worker_id_, shard);
  step_=modelproto_.step();
  // init params
  for(auto layer: train_net->layers())
    if(group_id_==0&&layer->locationid()==worker_id_)
      for(auto param: layer->GetParams()){
        if(param->owner()<0||param->owner()==param->id()){
          param->Init();
          Put(param, step_);
        }
        Get(param, step_);
      }
}

void Worker::Run(){
  step_=modelproto_.step();
  Performance perf(train_net_);
  try{
    while(!StopNow(step_)){
      RunOneBatch(step_, &perf);
      step_++;
    }
  }catch(WorkerException& e){
    LOG(ERROR)<<e.what();
  }
}
int Worker::Put(shared_ptr<Param> param, int step){
  auto msg=pmworker_->Put(param, step);
  if(msg!=nullptr)
    param_dealer_->Send(msg);
  return 1;
}
int Worker::Get(shared_ptr<Param> param, int step){
  if(param->version()<step){
    auto msg=pmworker_->Get(param, step);
    if(msg!=nullptr)
      param_dealer_->Send(msg);
  }
  return 1;
}
int Worker::Update(shared_ptr<Param> param, int step){
  auto msg=pmworker_->Update(param, step);
  if(msg!=nullptr)
    param_dealer_->Send(msg);
  return 1;
}
int Worker::Collect(shared_ptr<Param> param, int step){
  while(param->version()<step){
    Msg* msg=param_dealer_->Receive();
    if(msg==nullptr)
      return 0;
    pmworker_->Collect(&msg);
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
    Test(validation_net_, modelproto_.validation_steps(), perf!=nullptr);
  }
  if(TestNow(step)){
    LOG(ERROR)<<"Test at step "<<step;
    Test(test_net_, modelproto_.test_steps(), perf!=nullptr);
  }
  //tSyncData_=tSyncData; tSyncParam_=tSyncParam;

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
    if(layer->locationid()==worker_id_){
      if(layer->is_bridgedstlayer()){
        //auto* dst=static_cast<BridgeDstLayer*>(layer.get());
        // receive fea blobs
      }
      if(training){
        for(shared_ptr<Param> p: layer->GetParams()){
          if(Collect(p, step)==0){
            throw WorkerException();
          }
        }
      }
      layer->ComputeFeature(training);
      if(layer->is_bridgesrclayer()){
        // send fea blobs
      }
      if(training&&DisplayDebugInfo(step)&&layer->mutable_data()!=nullptr){
        LOG(INFO)<<StringPrintf("Forward layer  %10s data norm1 %13.9f",
            layer->name().c_str(), layer->data().asum_data());
      }
    }
  }
}

void BPWorker::Backward(shared_ptr<NeuralNet> net, int step){
  auto& layers=net->layers();
  for (auto it = layers.rbegin(); it != layers.rend(); it++){
    shared_ptr<Layer> layer=*it;
    if(layer->locationid()==worker_id_){
      if(layer->is_bridgesrclayer()){
        //auto* src=static_cast<BridgeSrcLayer*>(layer.get());
        // receive grad blobs
      }
      layer->ComputeGradient();
      if(DisplayDebugInfo(step)&&layer->mutable_grad()!=nullptr){
        LOG(INFO)<<StringPrintf("Backward layer %10s grad norm1 %13.9f\t",
            layer->name().c_str(), layer->grad().asum_data());
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
