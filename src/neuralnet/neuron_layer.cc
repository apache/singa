#include <glog/logging.h>
#include <algorithm>

#include "neuralnet/layer.h"
#include "utils/singleton.h"
#include "mshadow/tensor.h"
#include "mshadow/cxxnet_op.h"
namespace singa {

using namespace mshadow;
using mshadow::cpu;

using mshadow::Shape;
using mshadow::Shape1;
using mshadow::Shape2;
using mshadow::Shape3;
using mshadow::Shape4;
using mshadow::Tensor;

using std::string;
using std::vector;

inline Tensor<cpu, 4> Tensor4(Blob<float>* blob) {
  const vector<int>& shape = blob->shape();
  Tensor<cpu, 4> tensor(blob->mutable_cpu_data(),
      Shape4(shape[0], shape[1], shape[2], shape[3]));
  return tensor;
}

inline Tensor<cpu, 3> Tensor3(Blob<float>* blob) {
  const vector<int>& shape = blob->shape();
  Tensor<cpu, 3> tensor(blob->mutable_cpu_data(),
      Shape3(shape[0], shape[1], blob->count() / shape[0] / shape[1]));
  return tensor;
}

inline Tensor<cpu, 2> Tensor2(Blob<float>* blob) {
  const vector<int>& shape = blob->shape();
  Tensor<cpu, 2> tensor(blob->mutable_cpu_data(),
      Shape2(shape[0], blob->count() / shape[0]));
  return tensor;
}

inline Tensor<cpu, 1> Tensor1(Blob<float>* blob) {
  Tensor<cpu, 1> tensor(blob->mutable_cpu_data(), Shape1(blob->count()));
  return tensor;
}

/************ Implementation for ConvolutionLayer*************************/
ConvolutionLayer::~ConvolutionLayer() {
  delete weight_;
  delete bias_;
}
void ConvolutionLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  ConvolutionProto conv_conf = proto.convolution_conf();
  kernel_ = conv_conf.kernel();
  CHECK_GT(kernel_, 0) << "Filter size cannot be zero.";
  pad_ = conv_conf.pad();
  stride_ = conv_conf.stride();
  num_filters_ = conv_conf.num_filters();
  if (partition_dim() > 0)
    num_filters_ /= npartitions;
  const vector<int>& srcshape = srclayers_[0]->data(this).shape();
  int dim = srcshape.size();
  CHECK_GT(dim, 2);
  width_ = srcshape[dim - 1];
  height_ = srcshape[dim - 2];
  if (dim > 3)
    channels_ = srcshape[dim - 3];
  else if (dim > 2)
    channels_ = 1;
  batchsize_ = srcshape[0];
  conv_height_ = (height_ + 2 * pad_ - kernel_) / stride_ + 1;
  conv_width_ = (width_ + 2 * pad_ - kernel_) / stride_ + 1;
  col_height_ = channels_ * kernel_ * kernel_;
  col_width_ = conv_height_ * conv_width_;
  vector<int> shape{batchsize_, num_filters_, conv_height_, conv_width_};
  data_.Reshape(shape);
  grad_.Reshape(shape);
  col_data_.Reshape(vector<int>{col_height_, col_width_});
  col_grad_.Reshape(vector<int>{col_height_, col_width_});
  weight_ = Param::Create(proto.param(0));
  bias_ = Param::Create(proto.param(1));
  weight_->Setup(vector<int>{num_filters_, col_height_});
  bias_->Setup(vector<int>{num_filters_});
}

void ConvolutionLayer::ComputeFeature(int flag, Metric* perf) {
  auto src = Tensor4(srclayers_[0]->mutable_data(this));
  auto data = Tensor3(&data_);
  auto col = Tensor2(&col_data_);
  auto weight = Tensor2(weight_->mutable_data());
  auto bias = Tensor1(bias_->mutable_data());
  for (int n = 0; n < batchsize_; n++) {
    if (pad_ > 0)
      col = expr::unpack_patch2col(pad(src[n], pad_), kernel_, stride_);
    else
      col = expr::unpack_patch2col(src[n], kernel_, stride_);
    data[n] = dot(weight, col);
  }
  data += expr::broadcast<1>(bias, data.shape);
}

void ConvolutionLayer::ComputeGradient(int flag, Metric* perf) {
  auto src = Tensor4(srclayers_[0]->mutable_data(this));
  auto col = Tensor2(&col_data_);
  auto weight = Tensor2(weight_->mutable_data());
  auto grad = Tensor3(&grad_);
  auto gcol = Tensor2(&col_grad_);
  auto gweight = Tensor2(weight_->mutable_grad());
  auto gbias = Tensor1(bias_->mutable_grad());
  Blob<float>* gsrcblob = srclayers_[0]->mutable_grad(this);
  Tensor<cpu, 4> gsrc(nullptr, Shape4(batchsize_, channels_, height_, width_));
  if (gsrcblob != nullptr)
    gsrc.dptr = gsrcblob->mutable_cpu_data();
  gbias = expr::sumall_except_dim<1>(grad);
  gweight = 0.0f;
  Shape<3> padshp(gsrc.shape.SubShape());
  padshp[0] += 2 * pad_;
  padshp[1] += 2 * pad_;
  Shape<2> imgshp = Shape2(height_, width_);
  for (int n = 0; n < batchsize_; n++) {
    if (pad_ > 0)
      col = expr::unpack_patch2col(pad(src[n], pad_), kernel_, stride_);
    else
      col = expr::unpack_patch2col(src[n], kernel_, stride_);
    gweight += dot(grad[n], col.T());
    if (gsrcblob != nullptr) {
      gcol = dot(weight.T(), grad[n]);
      gsrc[n] = crop(expr::pack_col2patch(gcol, padshp, kernel_, stride_),
          imgshp);
    }
  }
}

/****************** Implementation for DropoutLayer ***********************/
void DropoutLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  data_.ReshapeLike(srclayers_[0]->data(this));
  grad_.ReshapeLike(*srclayers_[0]->mutable_grad(this));
  mask_.Reshape(srclayers_[0]->data(this).shape());
  pdrop_ = proto.dropout_conf().dropout_ratio();
}

void DropoutLayer::ComputeFeature(int flag, Metric* perf) {
  // check training
  if ((flag & kTrain) != kTrain) {
    data_.CopyFrom(srclayers_[0]->data(this));
    return;
  }
  float pkeep = 1 - pdrop_;
  auto mask = Tensor1(&mask_);
  mask = expr::F<op::threshold>(TSingleton<Random<cpu>>::Instance() \
                      ->uniform(mask.shape), pkeep) * (1.0f/pkeep);
  auto data = Tensor1(&data_);
  auto src = Tensor1(srclayers_[0]->mutable_data(this));
  data = src * mask;
}

void DropoutLayer::ComputeGradient(int flag, Metric* perf)  {
  auto mask = Tensor1(&mask_);
  auto grad = Tensor1(&grad_);
  auto gsrc = Tensor1(srclayers_[0]->mutable_grad(this));
  gsrc = grad * mask;
}


/**************** Implementation for RBMLayer********************/
Blob<float>* RBMLayer::Sample(int flag) {
  Tensor<cpu, 2> sample, data;
  if ((flag & kPositive) == kPositive || first_gibbs_) {
    data = Tensor2(&data_);
    sample = Tensor2(&sample_);
  } else {
    data = Tensor2(&neg_data_);
    sample = Tensor2(&neg_sample_);
  }
  auto random = TSingleton<Random<cpu>>::Instance();
  if (gaussian_) {
    random->SampleGaussian(sample, 0.0f, 1.0f);
    sample += data;
  } else {
    random->SampleBinary(sample, data);
  }
  return (flag & kPositive) == kPositive || first_gibbs_ ?
    &sample_ : &neg_sample_;
}
void RBMLayer::Setup(const LayerProto& proto, int npartitions) {
  CHECK_EQ(npartitions, 1);  //  TODO test for npartitions > 1
  Layer::Setup(proto, npartitions);
  hdim_ = proto.rbm_conf().hdim();
  gaussian_ = proto.rbm_conf().gaussian();
  first_gibbs_ = true;
}
/**************** Implementation for RBMVisLayer********************/
RBMVisLayer::~RBMVisLayer() {
  delete weight_;
  delete bias_;
}

void RBMVisLayer::Setup(const LayerProto& proto, int npartitions) {
  RBMLayer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 2);
  hid_layer_ = nullptr;
  for (auto src : srclayers_) {
    for (auto dst : src->srclayers()) {
      if (dst->name() == name()) {
        CHECK(hid_layer_ == nullptr);
        hid_layer_ = static_cast<RBMHidLayer*>(src);
      }
    }
  }
  input_layer_ = srclayers_[0] != hid_layer_ ? srclayers_[0]: srclayers_[1];
  const auto& src = input_layer_->data(this);
  batchsize_ = src.shape()[0];
  data_.ReshapeLike(src);
  neg_data_.ReshapeLike(data_);
  neg_sample_.ReshapeLike(data_);
  vdim_ = src.count() / batchsize_;
  weight_ = Param::Create(proto.param(0));
  weight_ ->Setup(vector<int>{hdim_, vdim_});
  bias_ = Param::Create(proto.param(1));
  bias_->Setup(vector<int>{vdim_});
}

void RBMVisLayer::ComputeFeature(int flag, Metric* perf) {
  if ((flag & kPositive) == kPositive) {
    data_.CopyFrom(input_layer_->data(this), true);
    first_gibbs_ = true;
  } else if ((flag & kNegative) == kNegative) {
    // fetch sampling results from hidden layer
    auto hid_sample = Tensor2(hid_layer_->Sample(flag));
    auto data = Tensor2(&neg_data_);
    auto weight = Tensor2(weight_->mutable_data());
    auto bias = Tensor1(bias_->mutable_data());
    data = dot(hid_sample, weight);
    data += expr::repmat(bias, batchsize_);
    data = expr::F<op::sigmoid>(data);
    if ((flag & kTest) == kTest) {
      const float *dptr = data_.cpu_data(), *rcns = neg_data_.cpu_data();
      float err = 0.f;
      for (int i = 0; i < data_.count(); i++) {
        err += (dptr[i] - rcns[i]) * (dptr[i] - rcns[i]);
      }
      perf->Add("Squared Error", err / batchsize_);
    }
    first_gibbs_ = false;
  }
}

void RBMVisLayer::ComputeGradient(int flag, Metric* perf) {
  auto vis_pos = Tensor2(&data_);
  auto vis_neg = Tensor2(&neg_data_);
  auto hid_pos = Tensor2(hid_layer_->mutable_data(this));
  auto hid_neg = Tensor2(hid_layer_->mutable_neg_data(this));

  auto gbias = Tensor1(bias_->mutable_grad());
  gbias = expr::sum_rows(vis_neg);
  gbias -= expr::sum_rows(vis_pos);
  gbias /= batchsize_;

  auto gweight = Tensor2(weight_->mutable_grad());
  gweight = dot(hid_neg.T(), vis_neg);
  gweight -= dot(hid_pos.T(), vis_pos);
  gweight /= batchsize_;
}
/**************** Implementation for RBMHidLayer********************/
RBMHidLayer::~RBMHidLayer() {
  delete weight_;
  delete bias_;
}

void RBMHidLayer::Setup(const LayerProto& proto,
      int npartitions) {
  RBMLayer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 1);
  const auto& src_data = srclayers_[0]->data(this);
  batchsize_ = src_data.shape()[0];
  vdim_ = src_data.count() / batchsize_;
  data_.Reshape(vector<int>{batchsize_, hdim_});
  neg_data_.ReshapeLike(data_);
  sample_.ReshapeLike(data_);
  neg_sample_.ReshapeLike(data_);
  weight_ = Param::Create(proto.param(0));
  weight_->Setup(vector<int>{hdim_, vdim_});
  bias_ = Param::Create(proto.param(1));
  bias_->Setup(vector<int>{hdim_});
  vis_layer_ = static_cast<RBMVisLayer*> (srclayers_[0]);
}

void RBMHidLayer::ComputeFeature(int flag, Metric* perf) {
  auto weight = Tensor2(weight_->mutable_data());
  auto bias = Tensor1(bias_->mutable_data());

  Tensor<cpu, 2> data, src;
  if ((flag & kPositive) == kPositive) {
    data = Tensor2(&data_);
    src = Tensor2(vis_layer_->mutable_data(this));
    first_gibbs_ = true;
  } else {
    data = Tensor2(&neg_data_);
    // hinton's science paper does not sample the vis layer
    src = Tensor2(vis_layer_->mutable_neg_data(this));
    first_gibbs_ = false;
  }
  data = dot(src, weight.T());
  data += expr::repmat(bias, batchsize_);

  if (!gaussian_)
    data = expr::F<op::sigmoid>(data);
}

void RBMHidLayer::ComputeGradient(int flag, Metric* perf) {
  auto hid_pos = Tensor2(&data_);
  auto hid_neg = Tensor2(&neg_data_);
  auto gbias = Tensor1(bias_->mutable_grad());
  gbias = expr::sum_rows(hid_neg);
  gbias -= expr::sum_rows(hid_pos);
  gbias /= batchsize_;
}
/*********** Implementation for InnerProductLayer**********/
InnerProductLayer::~InnerProductLayer() {
  delete weight_;
  delete bias_;
}

void InnerProductLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 1);
  const auto& src = srclayers_[0]->data(this);
  batchsize_ = src.shape()[0];
  vdim_ = src.count() / batchsize_;
  hdim_ = layer_proto_.innerproduct_conf().num_output();
  transpose_ = proto.innerproduct_conf().transpose();
  if (partition_dim() > 0)
    hdim_ /= npartitions;
  data_.Reshape(vector<int>{batchsize_, hdim_});
  grad_.ReshapeLike(data_);
  weight_ = Param::Create(proto.param(0));
  bias_ = Param::Create(proto.param(1));
  if (transpose_)
    weight_->Setup(vector<int>{vdim_, hdim_});
  else
    weight_->Setup(vector<int>{hdim_, vdim_});
  bias_->Setup(vector<int>{hdim_});
}

void InnerProductLayer::ComputeFeature(int flag, Metric* perf) {
  auto data = Tensor2(&data_);
  auto src = Tensor2(srclayers_[0]->mutable_data(this));
  auto weight = Tensor2(weight_->mutable_data());
  auto bias = Tensor1(bias_->mutable_data());
  if (transpose_)
    data = dot(src, weight);
  else
    data = dot(src, weight.T());
  // repmat: repeat bias vector into batchsize rows
  data += expr::repmat(bias, batchsize_);
}

void InnerProductLayer::ComputeGradient(int flag, Metric* perf) {
  auto src = Tensor2(srclayers_[0]->mutable_data(this));
  auto grad = Tensor2(&grad_);
  auto weight = Tensor2(weight_->mutable_data());
  auto gweight = Tensor2(weight_->mutable_grad());
  auto gbias = Tensor1(bias_->mutable_grad());

  gbias = expr::sum_rows(grad);
  if (transpose_)
    gweight = dot(src.T(), grad);
  else
    gweight = dot(grad.T(), src);
  if (srclayers_[0]->mutable_grad(this) != nullptr) {
    auto gsrc = Tensor2(srclayers_[0]->mutable_grad(this));
    if (transpose_)
      gsrc = dot(grad, weight.T());
    else
      gsrc = dot(grad, weight);
  }
}
/***************** Implementation for LRNLayer *************************/
void LRNLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 1);
  lsize_ = proto.lrn_conf().local_size();
  CHECK_EQ(lsize_ % 2, 1) << "LRN only supports odd values for Localvol";
  knorm_ = proto.lrn_conf().knorm();
  alpha_ = proto.lrn_conf().alpha();
  beta_ = proto.lrn_conf().beta();
  const vector<int>& s = srclayers_[0]->data(this).shape();
  data_.Reshape(s);
  grad_.Reshape(s);
  norm_.Reshape(s);
  batchsize_ = s[0];
  channels_ = s[1];
  height_ = s[2];
  width_ = s[3];
}

void LRNLayer::ComputeFeature(int flag, Metric* perf) {
  const float salpha = alpha_ / lsize_;
  auto src = Tensor4(srclayers_[0]->mutable_data(this));
  auto data = Tensor4(&data_);
  auto norm = Tensor4(&norm_);
  // stores normalizer without power
  norm = expr::chpool<red::sum>(expr::F<op::square>(src), lsize_) * salpha
    + knorm_;
  data = src * expr::F<op::power>(norm, -beta_);
}

void LRNLayer::ComputeGradient(int flag, Metric* perf) {
  const float salpha = alpha_ / lsize_;
  auto src = Tensor4(srclayers_[0]->mutable_data(this));
  auto norm = Tensor4(&norm_);
  auto grad = Tensor4(&grad_);
  auto gsrc = Tensor4(srclayers_[0]->mutable_grad(this));

  gsrc = grad * expr::F<op::power>(norm, -beta_);
  gsrc += (- 2.0f * beta_ * salpha) * expr::chpool<red::sum>(
      grad * src * expr::F<op::power>(norm, -beta_ - 1.0f), lsize_)  * src;
}

/******************** Implementation for PoolingLayer******************/
void PoolingLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 1);
  PoolingProto pool_conf = proto.pooling_conf();
  kernel_ = pool_conf.kernel();
  stride_ = pool_conf.stride();
  CHECK_LT(pad_, kernel_);
  pool_ = proto.pooling_conf().pool();
  CHECK(pool_ == PoolingProto_PoolMethod_AVE
        || pool_ == PoolingProto_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
  const auto& srcshape = srclayers_[0]->data(this).shape();
  int dim = srcshape.size();
  CHECK_GT(dim, 2);
  width_ = srcshape[dim - 1];
  height_ = srcshape[dim - 2];
  if (dim > 3)
    channels_ = srcshape[dim-3];
  else
    channels_ = 1;
  batchsize_ = srcshape[0];
  pooled_height_ = static_cast<int>((height_ - kernel_) / stride_) + 1;
  pooled_width_ = static_cast<int>((width_ - kernel_) / stride_) + 1;
  data_.Reshape(vector<int>{batchsize_, channels_, pooled_height_,
                            pooled_width_});
  grad_.ReshapeLike(data_);
}

void PoolingLayer::ComputeFeature(int flag, Metric* perf) {
  auto src = Tensor4(srclayers_[0]->mutable_data(this));
  auto data = Tensor4(&data_);
  if (pool_ == PoolingProto_PoolMethod_MAX)
    data = expr::pool<red::maximum>(src, kernel_, stride_);
  else if (pool_ == PoolingProto_PoolMethod_AVE)
    data = expr::pool<red::sum>(src, kernel_, stride_)
      * (1.0f / (kernel_ * kernel_));
}

/*
 * partition only on num/channel dim
 * assume grad and data have the same paritition
 */
void PoolingLayer::ComputeGradient(int flag, Metric* perf) {
  auto src = Tensor4(srclayers_[0]->mutable_data(this));
  auto gsrc = Tensor4(srclayers_[0]->mutable_grad(this));
  auto data = Tensor4(&data_);
  auto grad = Tensor4(&grad_);
  if (pool_ == PoolingProto_PoolMethod_MAX)
    gsrc = expr::unpool<red::maximum>(src, data, grad, kernel_, stride_);
  else if (pool_ == PoolingProto_PoolMethod_AVE)
    gsrc = expr::unpool<red::sum>(src, data, grad, kernel_, stride_)
           * (1.0f / (kernel_ * kernel_));
}

/***************** Implementation for ReLULayer *****************************/
void ReLULayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  data_.ReshapeLike(srclayers_[0]->data(this));
  grad_.ReshapeLike(*(srclayers_[0]->mutable_grad(this)));
}

void ReLULayer::ComputeFeature(int flag, Metric* perf) {
  auto data = Tensor1(&data_);
  auto src = Tensor1(srclayers_[0]->mutable_data(this));
  data = expr::F<op::relu>(src);
}

void ReLULayer::ComputeGradient(int flag, Metric* perf) {
  auto data = Tensor1(&data_);
  auto grad = Tensor1(&grad_);
  auto gsrc = Tensor1(srclayers_[0]->mutable_grad(this));
  gsrc = expr::F<op::relu_grad>(data)*grad;
}

/*******************Implementation of SigmoidLayer***************************/
void SigmoidLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  data_.ReshapeLike(srclayers_[0]->data(this));
  grad_.ReshapeLike(srclayers_[0]->grad(this));
}

void SigmoidLayer::ComputeFeature(int flag, Metric* perf) {
  auto data = Tensor1(&data_);
  auto src = Tensor1(srclayers_[0]->mutable_data(this));
  data = expr::F<op::sigmoid>(src);
}

void SigmoidLayer::ComputeGradient(int flag, Metric* perf) {
  auto data = Tensor1(&data_);
  auto grad = Tensor1(&grad_);
  auto gsrc = Tensor1(srclayers_[0]->mutable_grad(this));
  gsrc = expr::F<op::sigmoid_grad>(data) * grad;
}
/*******************Implementation of TanLayer***************************/
void STanhLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  data_.ReshapeLike(srclayers_[0]->data(this));
  grad_.ReshapeLike(srclayers_[0]->grad(this));
}

void STanhLayer::ComputeFeature(int flag, Metric* perf) {
  auto data = Tensor1(&data_);
  auto src = Tensor1(srclayers_[0]->mutable_data(this));
  data = expr::F<op::stanh>(src);
}

void STanhLayer::ComputeGradient(int flag, Metric* perf) {
  auto data = Tensor1(&data_);
  auto grad = Tensor1(&grad_);
  auto gsrc = Tensor1(srclayers_[0]->mutable_grad(this));
  gsrc = expr::F<op::stanh_grad>(data) * grad;
}
/********* Implementation for BridgeDstLayer **************/
void BridgeDstLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 1);
  data_.Reshape(srclayers_[0]->data(this).shape());
  grad_.ReshapeLike(data_);
}

}  // namespace singa
