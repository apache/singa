#include <glog/logging.h>
#include <cmath>
#include <chrono>
#include <random>
#include "utils/param.h"
#include "proto/cluster.pb.h"
#include "mshadow/tensor.h"
#include "utils/singleton.h"
using namespace mshadow;
using std::vector;
using std::string;
namespace singa {

Param::Param():data_(nullptr), slice_start_(0), num_slices_(0),
  num_pending_requests_(0),local_version_(-1){
}
void Param::Setup(const ParamProto& proto, const vector<int>& shape){
  data_=std::make_shared<Blob<float>>(shape);
  grad_.Reshape(shape);
  history_.Reshape(shape);
  proto_=proto;
}

void Param::AddSlice(int slice_id, int size){
  int offset=0;
  if(slice_size_.size()>0){
    //must be added in order
    CHECK_EQ(slice_start_+num_slices_, slice_id);
    offset=slice_offset_.back()+slice_size_.back();
  }
  else{
    slice_start_=slice_id;
    offset=0;
  }
  slice_offset_.push_back(offset);
  slice_size_.push_back(size);
  pending_get_.push_back(false);
  pending_update_.push_back(false);
  pending_put_.push_back(false);
  num_slices_++;
}

void Param::InitValues(int version){
  Tensor<cpu, 1> data(mutable_cpu_data(), Shape1(size()));
  auto random=TSingleton<Random<cpu>>::Instance();
  switch (proto_.init_method()) {
  case ParamProto::kConstant:
    data=proto_.value();
    break;
  case ParamProto::kUniform:
    random->SampleUniform(data, proto_.low(), proto_.high());
    if(proto_.value())
      data*= proto_.value();
    break;
    /*
  case ParamProto::kUniformSqrtFanIn:
    CHECK_GT(fan_in_,0);
    random->SampleUniform(data, proto_.low(), proto_.high());
    if(proto_.value())
      data*= proto_.value()/ sqrt(fan_in_ / 3.0f);
    break;
    */
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
  set_version(version);
}

/**************Message related functions********/
Msg* Param::GenPutMsg(bool copy, int idx){
  CHECK_LT(idx, num_slices_);
  Msg* msg=new Msg();
  msg->set_type(kPut);
  char buf[128];
  sprintf(buf, "%d %f %f", slice_size_[idx],
      learning_rate_multiplier(), weight_decay_multiplier());
  void *ptr=mutable_cpu_data()+slice_offset_[idx];
  if(copy){
    sprintf(buf+strlen(buf), " %p ", nullptr);
    msg->add_frame(buf, strlen(buf));
    msg->add_frame(ptr, slice_size_[idx]*sizeof(float));
  }else{
    sprintf(buf+strlen(buf), " %p ", ptr);
    msg->add_frame(buf, strlen(buf));
  }
  pending_put_[idx]=true;
  num_pending_requests_++;
	return msg;
}

Msg* Param::GenGetMsg(bool copy, int idx){
  CHECK_LT(idx, num_slices_);
  Msg* msg=new Msg();
  msg->set_type(kGet);
  char buf[8]; sprintf(buf, " %c ", copy);
  msg->add_frame(buf, sizeof(buf));
  pending_get_[idx]=true;
  num_pending_requests_++;
  return msg;
}

Msg* Param::GenUpdateMsg(bool copy, int idx){
  CHECK_LT(idx, num_slices_);
  Msg* msg=new Msg();
  msg->set_type(kUpdate);
  char buf[8]; sprintf(buf, " %c ", copy);
  msg->add_frame(buf, sizeof(buf));
  void* ptr=grad_.mutable_cpu_data()+slice_offset_[idx];
  if(copy)
    msg->add_frame(ptr, slice_size_[idx]*sizeof(float));
  else{ // to share values of grad blob
    char buf[32]; sprintf(buf, " %p ", ptr);
    msg->add_frame(buf, strlen(buf));
  }
  pending_update_[idx]=true;
  num_pending_requests_++;
  return msg;
}

Msg* Param::GenSyncMsg(bool copy, int v){
  Msg* msg=new Msg();
  msg->set_type(kSyncRequest);
  msg->set_target(id(), local_version());
  msg->add_frame(mutable_cpu_data(), size()*sizeof(float));
  return msg;
}

Msg* Param::HandlePutMsg(Msg** msg){
  int size;
  float lr, wc;
  float* ptr;
  sscanf(static_cast<char*>((*msg)->frame_data()),
      "%d %f %f %p ", &size, &lr, &wc, &ptr);
  proto_.set_learning_rate_multiplier(lr);
  proto_.set_weight_decay_multiplier(wc);
  vector<int> shape{size};
  Setup(shape);
  set_local_version((*msg)->target_second());
  set_version((*msg)->target_second());
  if(ptr==nullptr){
    CHECK((*msg)->next_frame());
    CHECK_EQ(size* sizeof(float), (*msg)->frame_size());
    memcpy(mutable_cpu_data(), (*msg)->frame_data(), size*sizeof(float));
  } else{
    data_->set_cpu_data(ptr);
  }
  DeleteMsg(msg);
  return nullptr;
}

Msg* Param::HandleGetMsg(Msg** msg){
  char copy; sscanf(static_cast<char*>((*msg)->frame_data()), " %c ", &copy);
  (*msg)->next_frame();
  if(copy)
    (*msg)->add_frame(mutable_cpu_data(), sizeof(float)*size());
  // else the mem space is shared among all worker and servers
  (*msg)->SwapAddr();
  (*msg)->set_type(kRGet);
  return *msg;
}

int Param::ParseUpdateMsg(Msg** msg){
  char copy; sscanf(static_cast<char*>((*msg)->frame_data()), " %c ", &copy);
  (*msg)->next_frame();
  if(copy){
    CHECK((*msg)->frame_size());
    memcpy(mutable_cpu_grad(), (*msg)->frame_data(),(*msg)->frame_size());
  }else {// use the same data field of the grad blob
    float* ptr=nullptr;
    sscanf(static_cast<char*>((*msg)->frame_data()), " %p ", &ptr);
    grad_.set_cpu_data(ptr);
  }
  DeleteMsg(msg);
  return copy;
}

Msg* Param::GenUpdateResponseMsg(bool copy){
  Msg* msg=new Msg();
  msg->set_type(kRUpdate);
  char buf[8]; sprintf(buf, " %c ", copy);
  msg->add_frame(buf, sizeof(buf));
  if(copy)
    msg->add_frame(mutable_cpu_data(), size()*sizeof(float));
  return msg;
}

Msg* Param::HandleSyncMsg(Msg** msg){
  DeleteMsg(msg);
  return nullptr;
}

<<<<<<< HEAD
int Param::ParseSyncResponseMsg(Msg** msg, int slice_idx){
  DeleteMsg(msg);
  return 1;
}

int Param::ParseGetResponseMsg(Msg **msg, int slice_idx){
  CHECK_EQ(pending_get_[slice_idx], true);
  pending_get_[slice_idx]=false;
  ParseResponseMsg(msg, slice_idx);
  return (--num_pending_requests_)%num_slices_==0;
}

int Param::ParseUpdateResponseMsg(Msg **msg, int slice_idx){
  CHECK_EQ(pending_update_[slice_idx], true);
  pending_update_[slice_idx]=false;
  ParseResponseMsg(msg, slice_idx);
  return (--num_pending_requests_)%num_slices_==0;
}

void Param::ParseResponseMsg(Msg** msg, int slice_idx){
  char copy; sscanf(static_cast<char*>((*msg)->frame_data()), " %c ", &copy);
  (*msg)->next_frame();
  if(copy){
    LOG(ERROR)<<"copy";
    CHECK((*msg)->frame_size());
    memcpy(mutable_cpu_data()+slice_offset_[slice_idx],
        (*msg)->frame_data(), (*msg)->frame_size());
  }
  DeleteMsg(msg);
}
}

// namespace singa
