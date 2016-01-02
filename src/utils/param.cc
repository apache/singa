/************************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/

#include "singa/utils/param.h"

#include <glog/logging.h>
#include <cmath>
#include <random>
#include <unordered_map>
#include "mshadow/tensor.h"
#include "singa/utils/factory.h"
#include "singa/utils/singleton.h"
#include "singa/utils/common.h"

namespace singa {

using mshadow::cpu;
using mshadow::Random;
using mshadow::Shape1;
using mshadow::Tensor;
using std::vector;
using std::string;

ParamGenerator* ParamGenerator::Create(const ParamGenProto& proto) {
  auto factory = Singleton<Factory<ParamGenerator>>::Instance();
  ParamGenerator * gen = nullptr;
  if (proto.has_user_type())
    gen = factory->Create(proto.user_type());
  else
    gen = factory->Create(proto.type());
  gen->Init(proto);
  return gen;
}

void ParamGenerator::Fill(Blob<float>* blob) {
  Tensor<cpu, 1> data(blob->mutable_cpu_data(), Shape1(blob->count()));
  data = proto_.value();
}

void GaussianGen::Fill(Blob<float>* blob) {
  Tensor<cpu, 1> data(blob->mutable_cpu_data(), Shape1(blob->count()));
  auto random = TSingleton<Random<cpu>>::Instance();
  random->SampleGaussian(data, proto_.mean(), proto_.std());
  if (proto_.value() != 1)
    data *= proto_.value();
}

void GaussianSqrtFanInGen::Fill(Blob<float>* blob) {
  // only valid for param matrix with num of cols as fan in
  CHECK_EQ(blob->shape().size(), 2);
  Tensor<cpu, 1> data(blob->mutable_cpu_data(), Shape1(blob->count()));
  GaussianGen::Fill(blob);
  data /= sqrt(blob->shape().at(1));
}

void UniformGen::Fill(Blob<float>* blob) {
  Tensor<cpu, 1> data(blob->mutable_cpu_data(), Shape1(blob->count()));
  auto random = TSingleton<Random<cpu>>::Instance();
  random->SampleUniform(data, proto_.low(), proto_.high());
  if (proto_.value() != 1)
    data *= proto_.value();
}

void UniformSqrtFanInGen::Fill(Blob<float>* blob) {
  // only valid for param matrix with num of cols as fan in
  CHECK_EQ(blob->shape().size(), 2);
  Tensor<cpu, 1> data(blob->mutable_cpu_data(), Shape1(blob->count()));
  UniformGen::Fill(blob);
  data /= sqrt(blob->shape().at(1) / 3.0f);
}

void UniformSqrtFanInOutGen::Fill(Blob<float>* blob) {
  // only valid for param matrix with num of cols as fan in
  CHECK_EQ(blob->shape().size(), 2);
  Tensor<cpu, 1> data(blob->mutable_cpu_data(), Shape1(blob->count()));
  UniformGen::Fill(blob);
  data /= sqrt(blob->shape()[0] + blob->shape()[1]);
}

/****************** Param functions *********************************/
Param* Param::Create(const ParamProto& proto) {
  Factory<Param>* factory = Singleton<Factory<Param>>::Instance();
  Param* p = nullptr;
  if (proto.has_user_type())
    p = factory->Create(proto.user_type());
  else
    p = factory->Create(proto.type());
  p->Init(proto);
  return p;
}

const vector<int> Param::ComputeSlices(int num, const vector<Param*>& params) {
  // collect sizes of unique Params
  std::vector<int> paramsize;
  for (auto param : params)
    if (param->id() == param->owner())
      paramsize.push_back(param->size());
  // slice into lcm pieces to achieve good load-balance for both intra-group
  // partition (among servers in a group) and inter-group partition (each group
  // is assgined a sub-set of slices)
  auto param_slice = Slice(num, paramsize);
  vector<int> slices;
  for (auto const vec : param_slice)
    for (int len : vec)
      slices.push_back(len);
  return slices;
}

void Param::SliceParams(int num, const vector<Param*>& params) {
  auto slices = ComputeSlices(num, params);
  // construct map from Param ID to its slices <slice id, len>
  std::unordered_map<int, vector<std::pair<int, int>>> paramid2slices;
  int slice_id = 0;
  auto it = slices.begin();
  for (auto param : params) {
    if (param->id() == param->owner()) {
      int len = 0;
      while (len < param->size() && it != slices.end()) {
        paramid2slices[param->id()].push_back(std::make_pair(slice_id++, *it));
        len += *it;
        it++;
      }
      CHECK_EQ(param->size(), len) << "length misamtch for ID=" << param->id();
    }
  }
  for (auto param : params) {
    for (auto entry : paramid2slices[param->owner()]) {
      param->AddSlice(entry.first, entry.second);
      LOG(INFO) << "param id " << param->id() << " owner=" << param->owner()
        << ", slice id = " << entry.first << ", size = " << entry.second;
    }
  }
}

void Param::Setup(const vector<int>& shape) {
  data_.Reshape(shape);
  grad_.Reshape(shape);
  history_.Reshape(shape);
}

void Param::InitValues() {
  InitValues(0);
}

void Param::InitValues(int version) {
  ParamGenerator* gen = ParamGenerator::Create(proto_.init());
  gen->Fill(&data_);
  set_version(version);
}

void Param::ShareDataFrom(Param* other, bool cpu_only) {
  if (this == other) {
    LOG(WARNING) << "No need to share Param with itself";
    return;
  }

  proto_.set_owner(other->owner());
  CHECK_EQ(data_.count(), other->data_.count());
  data_.ShareData(&(other->data_), cpu_only);
  if (grad_.count() == 0)
    grad_.Reshape(data_.shape());
  version_ = other->version_;
  last_version_ = other->last_version_;
  slice_start_ = other->slice_start_;
  num_slices_ = other->num_slices_;
  slice_offset_ = other->slice_offset_;
  slice_size_ = other->slice_size_;
  // change pending list size equal to slice size
  pending_get_.resize(other->pending_get_.size());
  pending_update_.resize(other->pending_update_.size());
}

void Param::ShareFrom(Param* other) {
  if (this == other) {
    LOG(WARNING) << "No need to share Param with itself";
    return;
  }

  ShareDataFrom(other, false);
  grad_.ShareData(&(other->grad_), false);
}

void Param::FromProto(const BlobProto& blob) {
  data_.FromProto(blob);
}

void Param::ToProto(BlobProto* blob) {
  data_.ToProto(blob);
}

void Param::AddSlice(int slice_id, int size) {
  int offset = 0;
  if (slice_size_.size() > 0) {
    // must be added in order
    CHECK_EQ(slice_start_ + num_slices_, slice_id);
    offset = slice_offset_.back() + slice_size_.back();
  } else {
    slice_start_ = slice_id;
    offset = 0;
  }
  slice_offset_.push_back(offset);
  slice_size_.push_back(size);
  pending_get_.push_back(false);
  pending_update_.push_back(false);
  num_slices_++;
}

Msg* Param::GenPutMsg(bool copy, int idx) {
  CHECK_LT(idx, num_slices_);
  Msg* msg = new Msg();
  msg->set_type(kPut);
  const void* ptr = data_.cpu_data() + slice_offset_[idx];
  const void* p = ptr;
  if (copy) p = nullptr;
  msg->AddFormatFrame("iffp", slice_size_[idx], lr_scale(), wd_scale(), p);
  if (copy) {
    msg->AddFrame(ptr, slice_size_[idx] * sizeof(float));
  }
  return msg;
}

Msg* Param::GenGetMsg(bool copy, int idx) {
  CHECK_LT(idx, num_slices_);
  Msg* msg = new Msg();
  msg->set_type(kGet);
  msg->AddFormatFrame("ip",  copy, data_.mutable_cpu_data()
      + slice_offset_[idx]);
  pending_get_[idx] = true;
  num_pending_requests_++;
  return msg;
}

Msg* Param::GenUpdateMsg(bool copy, int idx) {
  CHECK_LT(idx, num_slices_);
  Msg* msg = new Msg();
  msg->set_type(kUpdate);
  msg->AddFormatFrame("i", copy);
  const void* ptr = grad_.cpu_data() + slice_offset_[idx];
  if (copy) {
    msg->AddFrame(ptr, slice_size_[idx]*sizeof(float));
  } else {
    msg->AddFormatFrame("p", ptr);  // to share values of grad blob
  }

  pending_update_[idx] = true;
  num_pending_requests_++;
  return msg;
}

Msg* Param::GenSyncMsg(int offset, int size) {
  Msg* msg = new Msg();
  msg->set_type(kSyncRequest);
  msg->set_trgt(ParamTrgt(-1, id()), last_version());
  // always copy data because syn is between server groups in diff procs
  msg->AddFrame(mutable_cpu_data(), data_.count()*sizeof(float));
  return msg;
}

Msg* Param::HandlePutMsg(Msg** msg, bool reserve) {
  // TODO(wangsheng) remove the check later
  CHECK(reserve);
  int size;
  float lr, wc;
  float* ptr;
  (*msg)->ParseFormatFrame("iffp", &size, &lr, &wc, &ptr);
  ParamProto proto;
  proto.set_lr_scale(lr);
  proto.set_wd_scale(wc);
  vector<int> shape{size};
  Init(proto);
  Setup(shape);
  if (ptr == nullptr) {
    CHECK((*msg)->NextFrame());
    CHECK_EQ(size * sizeof(float), (*msg)->FrameSize());
    memcpy(mutable_cpu_data(), (*msg)->FrameData(), size * sizeof(float));
  } else {
    data_.set_cpu_data(ptr);
  }
  if (!reserve) DeleteMsg(msg);
  return nullptr;
}

Msg* Param::HandleGetMsg(Msg** msg, bool reserve) {
  // TODO(wangsheng) remove the check later
  CHECK(!reserve);
  int copy;
  float* ptr;
  (*msg)->ParseFormatFrame("ip", &copy, &ptr);
  if (copy) {
    (*msg)->AddFrame(mutable_cpu_data(), sizeof(float) * size());
  } else if (ptr != data_.cpu_data()) {
    // this case reflects following situation:
    // worker 0 and server are in the same process, while worker 1 is not.
    // worker 1 "put" data into server, so server need to allocate memory.
    // then worker 0 "get" data from server, so server need:
    //  1. copy the data to the worker0 provided space
    //  2. change its own pointer to that space in order to share memory
    // in this case, the server always points to last worker's space
    memcpy(ptr, data_.cpu_data(), sizeof(float) * size());
    data_.set_cpu_data(ptr);
  }
  // else the mem space is shared among all worker and servers
  Msg* ret = nullptr;
  if (reserve) {
    ret = new Msg(**msg);
  } else {
    // if not reserve the msg, we reuse it as return value
    ret = *msg;
    *msg = nullptr;
  }
  ret->SwapAddr();
  ret->set_type(kRGet);
  return ret;
}

void Param::ParseUpdateMsgs(const vector<Msg*>& msgs) {
  CHECK_GT(msgs.size(), 0);
  float* server_grad = nullptr;
  vector<float*> worker_grad;
  for (auto* msg : msgs) {
    int copy;
    msg->ParseFormatFrame("i", &copy);
    msg->NextFrame();
    float* ptr = nullptr;
    if (copy) {
      ptr = static_cast<float*>(msg->FrameData());
      CHECK_EQ(size() * sizeof(float), msg->FrameSize());
    } else {
      msg->ParseFormatFrame("p", &ptr);
      server_grad = ptr;
    }
    worker_grad.push_back(ptr);
  }
  if (server_grad == nullptr)
    server_grad = worker_grad.at(0);
  for (float* grad : worker_grad) {
    if (grad != server_grad) {
      // TODO(wangsh) think about optimize it later?
      for (int i = 0; i < size(); i++) {
        server_grad[i] += grad[i];
      }
    }
  }
  grad_.set_cpu_data(server_grad);
}

const vector<Msg*> Param::GenUpdateResponseMsgs(vector<Msg*>* msgs,
                                                bool reserve) {
  // TODO(wangsheng) remove the check later
  CHECK(!reserve);
  vector<Msg*> ret;
  for (Msg* msg : *msgs) {
    Msg* ptr = reserve ? new Msg(*msg) : msg;
    ptr->FirstFrame();
    ptr->SwapAddr();
    ptr->set_type(kRUpdate);
    int copy;
    ptr->ParseFormatFrame("i", &copy);
    if (copy) {
      ptr->NextFrame();
      CHECK_EQ(ptr->FrameSize(), sizeof(float) * size());
      memcpy(ptr->FrameData(), mutable_cpu_data(), ptr->FrameSize());
    }
    ret.push_back(ptr);
  }
  // if not reserved, we remove all pointers
  if (!reserve) msgs->clear();
  return ret;
}

Msg* Param::HandleSyncMsg(Msg** msg, bool reserve) {
  // TODO(wangwei) handle it later
  if (!reserve) DeleteMsg(msg);
  return nullptr;
}

int Param::ParseGetResponseMsg(Msg *msg, int slice_idx) {
  CHECK(pending_get_[slice_idx]) << slice_idx;
  pending_get_[slice_idx] = false;
  ParseResponseMsg(msg, slice_idx);
  return (--num_pending_requests_) % num_slices_ == 0;
}

int Param::ParseUpdateResponseMsg(Msg *msg, int slice_idx) {
  CHECK(pending_update_[slice_idx]) << id() << " " << slice_idx;
  pending_update_[slice_idx] = false;
  ParseResponseMsg(msg, slice_idx);
  return (--num_pending_requests_) % num_slices_ == 0;
}

int Param::ParseSyncResponseMsg(Msg* msg, int slice_idx) {
  // TODO(wangwei) handle it later
  return 1;
}

void Param::ParseResponseMsg(Msg* msg, int slice_idx) {
  int copy;
  msg->ParseFormatFrame("i", &copy);
  msg->NextFrame();
  if (copy) {
    CHECK_EQ(msg->FrameSize(), slice_size_[slice_idx] * sizeof(float));
    memcpy(mutable_cpu_data() + slice_offset_[slice_idx],
        msg->FrameData(), msg->FrameSize());
  }
  // LOG(ERROR)<<"parse response norm "<<data_->asum_data()<<" of "<<id();
}

/************************ParamEntry***************************/
ParamEntry::ParamEntry(int total, Param* p) {
  num_total = total;
  shares.push_back(p);
}

void ParamEntry::AddParam(bool local, Param* p) {
  num_local += local;
  num_total += 1;
  if (local) shares.push_back(p);
}

}  // namespace singa
