#include "neuralnet/connection_layer.h"

namespace singa {

using std::vector;

/************* Implementation for ConcateLayer ***********/
void ConcateLayer::Setup(const LayerProto& proto, int npartitions) {
  // CHECK_EQ(npartitions, 1);
  Layer::Setup(proto, npartitions);
  size_t concate_dim = proto.concate_conf().concate_dim();
  CHECK_GE(concate_dim, 0);
  CHECK_GT(srclayers_.size(), 1);
  vector<int> shape = srclayers_[0]->data(this).shape();
  for (size_t i = 1; i < srclayers_.size(); i++) {
    const vector<int>& srcshape = srclayers_[i]->data(this).shape();
    for (size_t j = 0; j < shape.size(); j++)
      if (j == concate_dim)
        shape[j] += srcshape[j];
      else
        CHECK_EQ(shape[j], srcshape[j]);
  }
  data_.Reshape(shape);
  grad_.Reshape(shape);
}

void ConcateLayer::ComputeFeature(int flag, Metric *perf) {
  LOG(FATAL) << "Not implemented for Concate Layer";
}

void ConcateLayer::ComputeGradient(int flag, Metric* perf) {
  LOG(FATAL) << "Not implemented for Concate Layer";
}

/************* Implementation for SliceLayer****************/
void SliceLayer::Setup(const LayerProto& proto, int npartitions) {
  /*
  Layer::Setup(proto, npartitions);
  slice_dim_ = proto.slice_conf().slice_dim();
  slice_num_ = npartitions;
  CHECK_GE(slice_dim_, 0);
  CHECK_EQ(slice_num_, dstlayers_.size());
  data_.Reshape(srclayers_[0]->data(this).shape());
  grad_.ReshapeLike(data_);
  datavec_.resize(slice_num_);
  gradvec_.resize(slice_num_);
  CHECK_EQ(data_.count() % slice_num_, 0);  // restrict equal slicing
  // LOG(ERROR)<<"slice dim "<<slice_dim<<" slice num "<<slice_num;
  for (int i = 0; i < slice_num_; i++) {
    vector<int> newshape(data_.shape());
    newshape[slice_dim_] = newshape[slice_dim_] / slice_num_ +
      ((i == slice_num_ - 1) ? newshape[slice_dim_] % slice_num_ : 0);
    datavec_[i].Reshape(newshape);
    gradvec_[i].Reshape(newshape);
    // LOG(ERROR)<<"slice "<<IntVecToString(newshape);
  }
  */
  LOG(FATAL) << "Not implemented";
}

void SliceLayer::ComputeFeature(int flag, Metric *perf) {
  /*
  CHECK_EQ(srclayers_.size(), 1);
  if (slice_dim_ == 0) {
    const auto& blob = srclayers_.at(0)->data(this);
    int size = blob.count() / slice_num_;
    for (int i = 0; i < slice_num_; i++) {
      float* dst = datavec_[i].mutable_cpu_data();
      const float* src = blob.cpu_data() + i * size;
      memcpy(dst, src, size*sizeof(float));
    }
  }
  */
  LOG(FATAL) << "Not implemented";
}

void SliceLayer::ComputeGradient(int flag, Metric* perf) {
  LOG(FATAL) << "Not implemented";
}

/*
int SliceLayer::SliceID(const Layer* layer) const {
  CHECK(layer != nullptr);
  for (size_t i = 0; i < datavec_.size(); i++) {
    // LOG(ERROR)<<"get slice "<<IntVecToString(shapes_[i]);
    if (dstlayers_[i] == layer)
      return i;
  }
  CHECK(false);
  return -1;
}*/

/************* Implementation for SplitLayer****************/
void SplitLayer::Setup(const LayerProto& proto, int npartitions) {
  // CHECK_EQ(npartitions, 1);
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 1);
  data_.Reshape(srclayers_[0]->data(this).shape());
  grad_.Reshape(srclayers_[0]->data(this).shape());
}

void SplitLayer::ComputeFeature(int flag, Metric *perf) {
  LOG(FATAL) << "Not implemented";
}

void SplitLayer::ComputeGradient(int flag, Metric* perf) {
  LOG(FATAL) << "Not implemented";
}
}  // namespace singa
