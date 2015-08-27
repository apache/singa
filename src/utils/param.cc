#include <glog/logging.h>
#include <cmath>
#include <chrono>
#include <random>
#include "utils/param.h"
#include "proto/job.pb.h"
#include "mshadow/tensor.h"
#include "utils/singleton.h"
#include "utils/factory.h"
namespace singa {
using namespace mshadow;
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

void ParamGenerator::Fill (Blob<float>* blob) {
  Tensor<cpu, 1> data(blob->mutable_cpu_data(), Shape1(blob->count()));
  data = proto_.value();
}
void GaussianGen::Fill (Blob<float>* blob) {
  Tensor<cpu, 1> data(blob->mutable_cpu_data(), Shape1(blob->count()));
  auto random = TSingleton<Random<cpu>>::Instance();
  random->SampleGaussian(data, proto_.mean(), proto_.std());
  if(proto_.value() != 1)
    data *= proto_.value();
}
void GaussianSqrtFanInGen::Fill (Blob<float>* blob) {
  // only valid for param matrix with num of cols as fan in
  CHECK_EQ(blob->shape().size(), 2);
  Tensor<cpu, 1> data(blob->mutable_cpu_data(), Shape1(blob->count()));
  GaussianGen::Fill(blob);
  data /= sqrt(blob->shape().at(1));
}

void UniformGen::Fill (Blob<float>* blob) {
  Tensor<cpu, 1> data(blob->mutable_cpu_data(), Shape1(blob->count()));
  auto random = TSingleton<Random<cpu>>::Instance();
  random->SampleUniform(data, proto_.low(), proto_.high());
  if(proto_.value() != 1)
    data *= proto_.value();
}

void UniformSqrtFanInGen::Fill (Blob<float>* blob) {
  // only valid for param matrix with num of cols as fan in
  CHECK_EQ(blob->shape().size(), 2);
  Tensor<cpu, 1> data(blob->mutable_cpu_data(), Shape1(blob->count()));
  UniformGen::Fill(blob);
  data /= sqrt(blob->shape().at(1) / 3.0f);
}

void UniformSqrtFanInOutGen::Fill (Blob<float>* blob) {
  // only valid for param matrix with num of cols as fan in
  CHECK_EQ(blob->shape().size(), 2);
  Tensor<cpu, 1> data(blob->mutable_cpu_data(), Shape1(blob->count()));
  UniformGen::Fill(blob);
  data /= sqrt(blob->shape()[0] + blob->shape()[1]);
}
/*****************Param***********************************/
Param* Param::Create(const ParamProto& proto) {
  Factory<Param>* factory=Singleton<Factory<Param>>::Instance();
  Param* p = nullptr;
  if (proto.has_user_type())
    p = factory->Create(proto.user_type());
  else
    p = factory->Create(proto.type());
  p->Init(proto);
  return p;
}

Param::Param():local_version_(-1), slice_start_(0), num_slices_(0),
  num_pending_requests_(0), data_(nullptr) {
}

void Param::Setup(const vector<int>& shape) {
  data_ = std::make_shared<Blob<float>>(shape);
  grad_.Reshape(shape);
  history_.Reshape(shape);
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
  pending_put_.push_back(false);
  num_slices_++;
}

void Param::InitValues(int version) {
  ParamGenerator* gen = ParamGenerator::Create(proto_.init());
  gen->Fill(data_.get());
  set_version(version);
}
void Param::FromProto(const BlobProto& blob) {
  data_->FromProto(blob);
}
void Param::ToProto(BlobProto* blob) {
  data_->ToProto(blob);
}

/**************Message related functions********/
Msg* Param::GenPutMsg(bool copy, int idx) {
  CHECK_LT(idx, num_slices_);
  Msg* msg = new Msg();
  msg->set_type(kPut);
  void *ptr = mutable_cpu_data() + slice_offset_[idx];
  void *p = ptr;
  if (copy) p = nullptr;
  msg->AddFormatFrame("iffp", slice_size_[idx],
      lr_scale(), wd_scale(), p);
  if (copy) {
    msg->AddFrame(ptr, slice_size_[idx] * sizeof(float));
  }
  // pending_put_[idx]=true;
  // num_pending_requests_++;
  return msg;
}

Msg* Param::GenGetMsg(bool copy, int idx) {
  CHECK_LT(idx, num_slices_);
  Msg* msg = new Msg();
  msg->set_type(kGet);
  msg->AddFormatFrame("ip",  copy, data_->cpu_data() + slice_offset_[idx]);
  pending_get_[idx] = true;
  num_pending_requests_++;
  return msg;
}

Msg* Param::GenUpdateMsg(bool copy, int idx) {
  CHECK_LT(idx, num_slices_);
  Msg* msg = new Msg();
  msg->set_type(kUpdate);
  msg->AddFormatFrame("i", copy);
  void* ptr = grad_.mutable_cpu_data() + slice_offset_[idx];
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
  msg->set_trgt(ParamTrgt(-1, id()), local_version());
  // always copy data because syn is between server groups in diff procs
  msg->AddFrame(mutable_cpu_data(), data_->count()*sizeof(float));
  return msg;
}

Msg* Param::HandlePutMsg(Msg** msg, bool reserve) {
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
    CHECK_EQ(size* sizeof(float), (*msg)->FrameSize());
    memcpy(mutable_cpu_data(), (*msg)->FrameData(), size*sizeof(float));
  } else {
    data_->set_cpu_data(ptr);
  }
  if (!reserve)
    DeleteMsg(msg);
  return nullptr;
}

Msg* Param::HandleGetMsg(Msg** msg, bool reserve) {
  int copy;
  float* ptr;
  (*msg)->ParseFormatFrame("ip", &copy, &ptr);
  if (copy) {
    (*msg)->AddFrame(mutable_cpu_data(), sizeof(float)*size());
  } else if (ptr != data_->cpu_data()) {
    memcpy(ptr, data_->cpu_data(), sizeof(float)*size());
    data_->set_cpu_data(ptr);
  }
  // else the mem space is shared among all worker and servers
  (*msg)->SwapAddr();
  (*msg)->set_type(kRGet);
  return *msg;
}

void Param::ParseUpdateMsgs(const vector<Msg*>& msgs) {
  CHECK_GT(msgs.size(), 0);
  float* server_grad = nullptr;
  vector<float*> worker_grad;
  for (auto *msg : msgs) {
    int copy;
    msg->ParseFormatFrame("i", &copy);
    msg->NextFrame();
    float* ptr = nullptr;
    if (copy) {
      ptr = static_cast<float*> (msg->FrameData());
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
      for (int i =0; i < size(); i++) {
        server_grad[i] += grad[i];
      }
    }
  }
  grad_.set_cpu_data(server_grad);
}

const vector<Msg*> Param::GenUpdateResponseMsgs(const vector<Msg*>& msgs) {
  vector<Msg*> ret;
  for (auto msg : msgs) {
    msg->FirstFrame();
    msg->SwapAddr();
    msg->set_type(kRUpdate);
    int copy;
    msg->ParseFormatFrame("i", &copy);
    if (copy) {
      msg->NextFrame();
      CHECK_EQ(msg->FrameSize(), sizeof(float) * size());
      memcpy(msg->FrameData(), mutable_cpu_data(), msg->FrameSize());
    }
    ret.push_back(msg);
  }
  return ret;
}

Msg* Param::HandleSyncMsg(Msg** msg, bool reserve) {
  if (!reserve)
    DeleteMsg(msg);
  return nullptr;
}

int Param::ParseSyncResponseMsg(Msg* msg, int slice_idx) {
  return 1;
}

int Param::ParseGetResponseMsg(Msg *msg, int slice_idx) {
  CHECK_EQ(pending_get_[slice_idx], true);
  pending_get_[slice_idx] = false;
  ParseResponseMsg(msg, slice_idx);
  return (--num_pending_requests_)%num_slices_ == 0;
}

int Param::ParseUpdateResponseMsg(Msg *msg, int slice_idx) {
  CHECK_EQ(pending_update_[slice_idx], true);
  pending_update_[slice_idx] = false;
  ParseResponseMsg(msg, slice_idx);
  return (--num_pending_requests_) % num_slices_ == 0;
}

void Param::ParseResponseMsg(Msg* msg, int slice_idx) {
  int copy;
  msg->ParseFormatFrame("i", &copy);
  msg->NextFrame();
  if (copy) {
    CHECK_EQ(msg->FrameSize(), slice_size_[slice_idx]*sizeof(float));
    memcpy(mutable_cpu_data()+slice_offset_[slice_idx],
        msg->FrameData(), msg->FrameSize());
  }
  // LOG(ERROR)<<"parse response norm "<<data_->asum_data()<<" of "<<id();
}

void Param::ShareFrom(const Param& other) {
  proto_.set_owner(other.owner());
  if (data_ != nullptr) {
    CHECK(std::equal(data_->shape().begin(), data_->shape().end(),
          other.data_->shape().begin()));
  }
  data_ = other.data_;
  if (grad_.count() == 0)
    grad_.Reshape(data_->shape());
  slice_offset_ = other.slice_offset_;
  slice_size_ = other.slice_size_;
  slice_start_ = other.slice_start_;
  num_slices_ = other.num_slices_;
  pending_get_ = other.pending_get_;
  pending_put_ = other.pending_put_;
  pending_update_ = other.pending_update_;
}

/************************ParamEntry***************************/
ParamEntry::ParamEntry():
  num_update(0), next_version(-1), num_local(0), num_total(0) {
}

ParamEntry::ParamEntry(int total, Param* p) : num_update(0), num_total(total) {
  shares.push_back(p);
}
void ParamEntry::AddParam(bool local, Param* p) {
  num_local += local;
  num_total += 1;
  if (local)
    shares.push_back(p);
}
}  // namespace singa
