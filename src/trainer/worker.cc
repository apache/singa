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
using std::thread;
namespace singa {
Worker::Worker(int thread_id, int group_id, int worker_id):
  thread_id_(thread_id), group_id_(group_id), worker_id_(worker_id){
  }

void Worker::Setup(const ModelProto& model,
    shared_ptr<NeuralNet> train_net){
  train_net_=train_net;
  modelproto_=model;
}

void Worker::Run(){
  param_dealer_=make_shared<Dealer>(2*thread_id_);
  param_dealer_->Connect(kInprocRouterEndpoint);
  param_poller_.Add(param_dealer_.get());
  layer_dealer_=make_shared<Dealer>(2*thread_id_+1);
  layer_dealer_->Connect(kInprocRouterEndpoint);

  { // TODO remove waiting pong msg
  Msg* ping=new Msg();
  ping->set_src(group_id_, worker_id_, kWorkerParam);
  ping->set_dst(-1,-1,kStub);
  ping->set_type(kConnect);
  ping->add_frame("PING", 4);
  param_dealer_->Send(&ping);
  ping=param_dealer_->Receive();
  string pong((char*)ping->frame_data(), ping->frame_size());
  CHECK_STREQ("PONG", pong.c_str());
  delete ping;
  }

  {
  Msg* ping=new Msg();
  ping->set_src(group_id_, worker_id_, kWorkerLayer);
  ping->set_dst(-1,-1,kStub);
  ping->set_type(kConnect);
  ping->add_frame("PING", 4);
  layer_dealer_->Send(&ping);
  ping=layer_dealer_->Receive();
  string pong((char*)ping->frame_data(), ping->frame_size());
  CHECK_STREQ("PONG", pong.c_str());
  delete ping;
  }
  step_=modelproto_.step();
  //layer_dealer_=std::make_shared<Dealer>(thread_id_*2);
  // init params
  for(auto layer: train_net_->layers()){
    //LOG(ERROR)<<layer->partitionid()<<" : "<<layer->name();
    if(layer->partitionid()==worker_id_)
      for(auto param: layer->GetParams()){
        if(group_id_==0){
          if(param->owner()==param->id()){
            param->Init(0);
            Put(param, step_);
          }else{
            Get(param, 0);
          }
        }else{
          Get(param, modelproto_.warmup_steps());
        }
      }
  }
  Metric perf;
  if(group_id_==0&&step_<modelproto_.warmup_steps()){
    for(step_=0;step_<modelproto_.warmup_steps();step_++)
      RunOneBatch(step_, &perf);
    for(auto layer: train_net_->layers()){
      //LOG(ERROR)<<layer->partitionid()<<" : "<<layer->name();
      if(layer->partitionid()==worker_id_)
        for(auto param: layer->GetParams())
          if(param->owner()==param->id())
            Put(param, step_);
    }
  }
  while(!StopNow(step_)){
    RunOneBatch(step_, &perf);
    step_++;
  }
}
int Worker::Put(shared_ptr<Param> param, int step){
  Msg* msg=new Msg();
  msg->set_src(group_id_, worker_id_, kWorkerParam);
  msg->set_dst(-1, -1, kStub);
  msg->set_type(kPut);
  msg->set_target(param->owner(), step);
  param_dealer_->Send(&msg);
  return 1;
}
int Worker::Get(shared_ptr<Param> param, int step){
  Msg* msg=new Msg();
  msg->set_src(group_id_, worker_id_, kWorkerParam);
  msg->set_dst(-1, -1, kStub);
  msg->set_type(kGet);
  msg->set_target(param->owner(), step);
  param_dealer_->Send(&msg);
  return 1;
}
int Worker::Update(shared_ptr<Param> param, int step){
  Msg* msg=new Msg();
  msg->set_src(group_id_, worker_id_, kWorkerParam);
  msg->set_dst(-1, -1, kStub);
  msg->set_type(kUpdate);
  msg->set_target(param->owner(), step);
  param_dealer_->Send(&msg);
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
    std::this_thread::sleep_for(std::chrono::milliseconds(kCollectSleepTime));
  }
  return 1;
}
const void Worker::DisplayPerformance(const Metric & perf, const string& prefix){
  /* TODO send perf to Stub thread for printing
     Msg* msg=new Msg();
     msg->set_src(group_id_, worker_id_, kWorkerParam);
     msg->set_dst(-1,-1, kStub);
     msg->set_type(kMetric);
     const string disp=perf.ToString();
     msg->AddFrame(disp.c_str(), disp.length());
     param_dealer_->Send(&msg);
     */
  LOG(ERROR)<<prefix<<" "<<perf.ToString();
}

void Worker::RunOneBatch(int step, Metric* perf){
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
  TrainOneBatch(step);
  if(perf!=nullptr){
    auto losslayers=train_net_->losslayers();
    for(auto layer: losslayers){
      if(layer->partitionid()==worker_id_){
        const float * ptr=layer->metric().cpu_data();
        for(int j=0;j<layer->metric().count();j++)
          perf->AddMetric(layer->name()+"-"+std::to_string(j), ptr[j]);
      }
    }
    perf->Inc();
    if(DisplayNow(step)){
      perf->Avg();
      DisplayPerformance(*perf, "Train at step "+std::to_string(step));
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

void Worker::Test(shared_ptr<NeuralNet> net, int nsteps, bool disperf){
  const auto& losslayers=net->losslayers();
  Metric perf;
  for(int step=0;step<nsteps;step++){
    TestOneBatch(net, step, kTest);
    if(disperf){
      for(auto layer: losslayers){
        if(layer->partitionid()==worker_id_){
          const float * ptr=layer->metric().cpu_data();
          for(int j=0;j<layer->metric().count();j++)
            perf.AddMetric(layer->name()+"-"+std::to_string(j), ptr[j]);
        }
      }
      perf.Inc();
    }
  }
  if(disperf){
    perf.Avg();
    DisplayPerformance(perf, "Test");
    perf.Reset();
  }
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
          CHECK_EQ(msg->src_first(), group_id_);
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
        layer_dealer_->Send(&msg);
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

}  // namespace singa
