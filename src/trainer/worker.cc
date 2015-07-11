#include <glog/logging.h>
#include <thread>
#include <memory>
#include <iostream>
#include <chrono>
#include <thread>
#include "utils/singleton.h"
#include "utils/factory.h"
#include "trainer/worker.h"
#include "proto/model.pb.h"
namespace singa {
using std::thread;
using std::make_shared;

Worker::Worker(int thread_id, int group_id, int worker_id):
  thread_id_(thread_id), group_id_(group_id), worker_id_(worker_id){
}

void Worker::Setup(const ModelProto& model,
    shared_ptr<NeuralNet> train_net){
  train_net_=train_net;
  modelproto_=model;
  auto cluster=Cluster::Get();
  if(!(cluster->nserver_groups()&&cluster->server_update())){
    updater_=shared_ptr<Updater>(Singleton<Factory<Updater>>::Instance()
        ->Create("Updater"));
    updater_->Init(model.updater());
  }
}

void Worker::ConnectStub(shared_ptr<Dealer> dealer, EntityType type){
  if(updater_==nullptr){
    auto cluster=Cluster::Get();
    int sgid=group_id_/cluster->nworker_groups_per_server_group();
    CHECK(cluster->runtime()->JoinSGroup(group_id_, worker_id_, sgid));
  }

  dealer->Connect(kInprocRouterEndpoint);
  Msg* ping=new Msg();
  ping->set_src(group_id_, worker_id_, type);
  ping->set_dst(-1,-1,kStub);
  ping->set_type(kConnect);
  ping->add_frame("PING", 4);
  dealer->Send(&ping);
  ping=dealer->Receive();
  string pong((char*)ping->frame_data(), ping->frame_size());
  CHECK_STREQ("PONG", pong.c_str());
  delete ping;
}

void Worker::Run(){
  LOG(ERROR)<<"Worker (group_id = "<<group_id_
    <<", id = "<<worker_id_<<") starts";
  dealer_=make_shared<Dealer>(2*thread_id_);
  ConnectStub(dealer_, kWorkerParam);
  for(auto layer: train_net_->layers())
    if(layer->partition_id()==worker_id_)
      if(layer->is_bridgedstlayer()||layer->is_bridgesrclayer()){
        layer_dealer_=make_shared<Dealer>(2*thread_id_+1);
        ConnectStub(layer_dealer_, kWorkerLayer);
        break;
      }
  step_=modelproto_.step();
  // init params
  for(auto layer: train_net_->layers()){
    if(layer->partition_id()==worker_id_)
      for(auto param: layer->GetParams()){
        // only owners fill the memory of parameter values.
        // others share the memory with owners hence do not need to put/get.
        if(param->owner() == param->id()){
          if(group_id_%Cluster::Get()->nworker_groups_per_server_group()==0)
            param->InitValues(0);
          else{
            Get(param, modelproto_.warmup_steps());
          }
        }
      }
  }
  Metric perf;
  if(group_id_%Cluster::Get()->nworker_groups_per_server_group()==0){
    for(step_=0;step_<modelproto_.warmup_steps();step_++)
      RunOneBatch(step_, &perf);
    for(auto layer: train_net_->layers()){
      if(layer->partition_id()==worker_id_)
        for(auto param: layer->GetParams())
          if(param->owner()==param->id())
            Put(param, step_);
    }
  }
  while(!StopNow(step_)){
    RunOneBatch(step_, &perf);
    step_++;
  }

  Stop();
  LOG(ERROR)<<"Worker (group_id = "<<group_id_
    <<", id = "<<worker_id_<<") stops";
}

void Worker::Stop(){
  auto cluster=Cluster::Get();
  if(updater_ == nullptr){
    int sgid=group_id_/cluster->nworker_groups_per_server_group();
    cluster->runtime()->LeaveSGroup(group_id_, worker_id_, sgid);
  }
  Msg* msg=new Msg();
  msg->set_src(group_id_, worker_id_, kWorkerParam);
  msg->set_dst(-1,-1, kStub);
  msg->set_type(kStop);
  dealer_->Send(&msg); // use param dealer to send the stop msg
}
int Worker::Put(Param* param, int step){
  Msg* msg=new Msg();
  msg->set_src(group_id_, worker_id_, kWorkerParam);
  msg->set_dst(-1, -1, kStub);
  msg->set_type(kPut);
  msg->set_trgt(param->owner(), 0, step);
  dealer_->Send(&msg);
  return 1;
}
int Worker::Get(Param* param, int step){
  Msg* msg=new Msg();
  msg->set_src(group_id_, worker_id_, kWorkerParam);
  msg->set_dst(-1, -1, kStub);
  msg->set_type(kGet);
  msg->set_trgt(param->owner(), 0, step);
  dealer_->Send(&msg);
  return 1;
}
int Worker::Update(Param* param, int step){
  param->set_local_version(param->version());
  if(updater_){
    updater_->Update(step, param);
    param->set_version(param->version()+1);
  }else{
    Msg* msg=new Msg();
    msg->set_src(group_id_, worker_id_, kWorkerParam);
    msg->set_dst(-1, -1, kStub);
    msg->set_type(kUpdate);
    msg->set_trgt(param->owner(), 0, step);
    dealer_->Send(&msg);
  }
  return 1;
}

int Worker::CollectAll(shared_ptr<NeuralNet> net, int step){
  auto& layers=net->layers();
  for(auto& layer: layers){
    if(layer->partition_id()==worker_id_)
      for(Param* p: layer->GetParams()){
        Collect(p, step);
      }
  }
  return 1;
}
int Worker::Collect(Param* param, int step){
  while(param->version()<=param->local_version()){
    std::this_thread::sleep_for(std::chrono::milliseconds(kCollectSleepTime));
  }
  return 1;
}
void Worker::DisplayPerformance(const string& prefix, const Metric & perf) {
  Msg* msg=new Msg();
  msg->set_src(group_id_, worker_id_, kWorkerParam);
  msg->set_dst(-1,-1, kStub);
  msg->set_type(kMetric);
  msg->set_trgt(step_,0,0);
  msg->add_frame(prefix.c_str(), prefix.length());
  const string disp = perf.ToString();
  msg->add_frame(disp.c_str(), disp.length());
  dealer_->Send(&msg);
}

void Worker::RunOneBatch(int step, Metric* perf){
  if(ValidateNow(step)){
    //LOG(ERROR)<<"Validation at step "<<step;
    CollectAll(validation_net_, step);
    Test(modelproto_.validation_steps(),kValidation, validation_net_);
  }
  if(TestNow(step)){
    //LOG(ERROR)<<"Test at step "<<step;
    CollectAll(test_net_, step);
    Test(modelproto_.test_steps(), kTest, test_net_);
  }
  TrainOneBatch(step, perf);
  //LOG(ERROR)<<"Train "<<step;
  if(perf!=nullptr){
    if(DisplayNow(step)){
      DisplayPerformance("Train", *perf);
      perf->Reset();
    }
  }
  /*
  if(CheckpointNow(step)){
    pm_->Checkpoint(cluster_->workspace()+"/snapshot-"+std::to_string(step));
  }
  */
}

void Worker::ReceiveBlobs(shared_ptr<NeuralNet> net){
}

void Worker::SendBlob(){
}

void Worker::Test(int nsteps, Phase phase, shared_ptr<NeuralNet> net){
  Metric perf;
  for(int step=0;step<nsteps;step++){
    TestOneBatch(step, phase, net, &perf);
  }
  //perf.Avg();
  if(phase==kValidation)
    DisplayPerformance("Validation", perf);
  else if (phase==kTest)
    DisplayPerformance("Test", perf);
}

/****************************BPWorker**********************************/

BPWorker::BPWorker(int thread_id, int group_id, int worker_id):
  Worker(thread_id, group_id, worker_id){
}

void BPWorker::Forward(int step, Phase phase, shared_ptr<NeuralNet> net,
    Metric* perf){
  auto& layers=net->layers();
  for(auto& layer: layers){
    if(layer->partition_id()==worker_id_){
      if(layer->is_bridgedstlayer()){
        auto* dst=static_cast<BridgeDstLayer*>(layer);
        while(!dst->ready()){
          auto msg=layer_dealer_->Receive();
          CHECK_EQ(msg->src_first(), group_id_);
          string name((char*)msg->frame_data(), msg->frame_size());
          auto tmp=net->name2layer(name);
          CHECK(tmp->is_bridgedstlayer());
          auto* dstlayer=static_cast<BridgeDstLayer*>(tmp);
          auto data=dstlayer->mutable_data(nullptr);
          msg->next_frame();
          memcpy(data->mutable_cpu_data(), msg->frame_data(), msg->frame_size());
          dstlayer->set_ready(true);
          delete msg;
        }
      }
      if(phase==kTrain){
        for(Param* p: layer->GetParams()){
          Collect(p, step);
        }
      }
      //clock_t s=clock();
      layer->ComputeFeature(phase, perf);
      //LOG(ERROR)<<layer->name()<<":"<<(clock()-s)*1.0/CLOCKS_PER_SEC;
      if(layer->is_bridgesrclayer()){
        auto dst=layer->dstlayers().at(0);
        Msg *msg=new Msg();
        msg->set_src(group_id_, worker_id_, kWorkerLayer);
        msg->set_dst(group_id_, dst->partition_id(), kWorkerLayer);
        msg->add_frame(dst->name().c_str(), dst->name().length());
        auto const & blob=layer->data(nullptr);
        msg->add_frame(blob.cpu_data(), blob.count()*sizeof(float));
        layer_dealer_->Send(&msg);
      }
      if(phase == kTrain && DisplayDebugInfo(step))
        LOG(INFO) << layer->DebugString(step, kForward);
    }
  }
}

void BPWorker::Backward(int step, shared_ptr<NeuralNet> net){
  auto& layers=net->layers();
  for (auto it = layers.rbegin(); it != layers.rend(); it++){
    Layer* layer=*it;
    if(layer->partition_id()==worker_id_){
      if(layer->is_bridgesrclayer()){
        //auto* src=static_cast<BridgeSrcLayer*>(layer.get());
        // receive grad blobs
      }
      layer->ComputeGradient(kTrain);
      if(DisplayDebugInfo(step))
        LOG(INFO) << layer->DebugString(step, kBackward);
      for(Param* p: layer->GetParams())
        Update(p, step);
      if(layer->is_bridgedstlayer()){
        // send grad blobs
      }
    }
  }
}

void BPWorker::TrainOneBatch(int step, Metric* perf){
  Forward(step, kTrain, train_net_, perf);
  Backward(step, train_net_);
  auto losslayers=train_net_->losslayers();
}

void BPWorker::TestOneBatch(int step, Phase phase,
    shared_ptr<NeuralNet> net, Metric* perf){
  Forward(step, phase, net, perf);
}

}  // namespace singa
