#include <glog/logging.h>
#include <cmath>
#include <chrono>
#include <random>
#include "utils/param.h"
#include "mshadow/tensor.h"
#include "utils/singleton.h"
using namespace mshadow;
using std::vector;
using std::string;
namespace singa {

Param::Param():data_(nullptr), local_version_(-1){}

Msg* Param::GenPutMsg(bool copy, int v){
  Msg* msg=new Msg();
  msg->set_type(kPut);
  msg->set_target(owner(), version());
  char buf[128];
  sprintf(buf, "%d %f %f", size(),
      learning_rate_multiplier(), weight_decay_multiplier());
  if(copy){
    sprintf(buf+strlen(buf), " %p", nullptr);
    msg->add_frame(buf, strlen(buf));
    msg->add_frame(mutable_cpu_data(), size()*sizeof(float));
  }else{
    //share the data blob which includes the blob version
    sprintf(buf+strlen(buf), " %p", data_.get());
    msg->add_frame(buf, strlen(buf));
  }
	return msg;
}

Msg* Param::GenGetMsg(bool copy, int v){
  Msg* msg=new Msg();
  msg->set_type(kGet);
  msg->set_target(owner(), local_version()+1);
  msg->add_frame(&copy, sizeof(bool));
  return msg;
}

Msg* Param::GenUpdateMsg(bool copy, int v){
  Msg* msg=new Msg();
  msg->set_type(kUpdate);
  msg->set_target(owner(), v);
  msg->add_frame(&copy, sizeof(bool));
  if(copy)
    msg->add_frame(mutable_cpu_grad(), size()*sizeof(float));
  else{ // to share values of grad blob
    char buf[32]; sprintf(buf, "%p", &grad_);
    msg->add_frame(buf, strlen(buf));
    //LOG(ERROR)<<"param id="<<id()<<" ptr="<<buf;
  }
  return msg;
}

Msg* Param::GenSyncMsg(bool copy, int v){
  return nullptr;
}

Msg* Param::HandlePutMsg(Msg** msg){
  int size;
  float lr, wc;
  void* ptr;
  sscanf(static_cast<char*>((*msg)->frame_data()), "%d %f %f %p",
      &size, &lr, &wc, &ptr);
  proto_.set_learning_rate_multiplier(lr);
  proto_.set_weight_decay_multiplier(wc);
  vector<int> shape{size};
  grad_.Reshape(shape);
  history_.Reshape(shape);
  data_=std::make_shared<Blob<float>>(shape);
  if(ptr==nullptr){
    data_->set_version((*msg)->target_second());
    CHECK((*msg)->next_frame());
    CHECK_EQ(size* sizeof(float), (*msg)->frame_size());
    memcpy(mutable_cpu_data(), (*msg)->frame_data(), size*sizeof(float));
  } else{
    data_->ShareData(*static_cast<Blob<float>*>(ptr));
  }
  DeleteMsg(msg);
  return nullptr;
}

Msg* Param::HandleGetMsg(Msg** msg){
  if((*msg)->target_second()<=version()){
    bool* copy=static_cast<bool*>((*msg)->frame_data());
    (*msg)->next_frame();
    if(*copy)
      (*msg)->add_frame(mutable_cpu_data(), sizeof(float)*size());
    (*msg)->SwapAddr();
    (*msg)->set_type(kRGet);
  }
  return *msg;
}

const std::pair<bool, int> Param::ParseUpdateMsg(Msg** msg){
  int step=(*msg)->target_second();
  bool* copy=static_cast<bool*>((*msg)->frame_data());
  (*msg)->next_frame();
  if(*copy){
    CHECK((*msg)->frame_size());
    memcpy(mutable_cpu_grad(), (*msg)->frame_data(),(*msg)->frame_size());
  }else {// use the same data field of the grad blob
    Blob<float>* ptr=nullptr;
    sscanf(static_cast<char*>((*msg)->frame_data()), "%p", &ptr);
    //LOG(ERROR)<<"id="<<id()<<" ptr="<<ptr;
    grad_.ShareData(*ptr);
  }
  DeleteMsg(msg);
  return std::make_pair(*copy, step);
}

Msg* Param::GenUpdateResponseMsg(bool copy, int v){
  Msg* msg=new Msg();
  msg->set_type(kRUpdate);
  msg->set_target(owner(), v);
  msg->add_frame(&copy, sizeof(bool));
  if(copy)
    msg->add_frame(mutable_cpu_data(), size()*sizeof(float));
  return msg;
}

Msg* Param::HandleSyncMsg(Msg** msg){
  DeleteMsg(msg);
  return nullptr;
}

int Param::ParseSyncResponseMsg(Msg** msg){
  DeleteMsg(msg);
  return 1;
}
int Param::ParsePutResponseMsg(Msg **msg){
  return ParseSyncResponseMsg(msg);
}
int Param::ParseGetResponseMsg(Msg **msg){
  bool *copy=static_cast<bool*>((*msg)->frame_data());
  (*msg)->next_frame();
  if(*copy){
    CHECK((*msg)->frame_size());
    memcpy(mutable_cpu_data(), (*msg)->frame_data(), (*msg)->frame_size());
  }  // must be set after all other settings are done!
  set_version((*msg)->target_second());
  DeleteMsg(msg);
  return 1;
}
int Param::ParseUpdateResponseMsg(Msg **msg){
  return ParseGetResponseMsg(msg);
}

void Param::Setup(const ParamProto& proto, const vector<int>& shape,
    int fan_in){
  data_=std::make_shared<Blob<float>>(shape);
  grad_.Reshape(shape);
  history_.Reshape(shape);
  proto_=proto;
  fan_in_=fan_in;
}

void Param::Init(int v){
  Tensor<cpu, 1> data(mutable_cpu_data(), Shape1(size()));
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  auto random=ASingleton<Random<cpu>>::Instance(seed);
  switch (proto_.init_method()) {
  case ParamProto::kConstant:
    data=proto_.value();
    break;
  case ParamProto::kUniform:
    random->SampleUniform(data, proto_.low(), proto_.high());
    if(proto_.value())
      data*= proto_.value();
    break;
  case ParamProto::kUniformSqrtFanIn:
    CHECK_GT(fan_in_,0);
    random->SampleUniform(data, proto_.low(), proto_.high());
    if(proto_.value())
      data*= proto_.value()/ sqrt(fan_in_ / 3.0f);
    break;
  case ParamProto::kUniformSqrtFanInOut:
    random->SampleUniform(data, proto_.low(), proto_.high());
    if(proto_.value())
      data*= proto_.value()/ sqrt(data_->shape()[0] +data_->shape()[1]);
    break;
  case ParamProto::kGaussian:
    random->SampleGaussian(data, proto_.mean(), proto_.std());
    if(proto_.value())
      data*= proto_.value();
    break;
  case ParamProto::kGaussainSqrtFanIn:
    random->SampleGaussian(data, proto_.mean(), proto_.std());
    if(proto_.value())
      data*= proto_.value()/ sqrt(data_->shape()[0]);
    break;
  default:
    LOG(ERROR) << "Illegal parameter init method ";
    break;
  }
  set_version(v);
}
}  // namespace singa
