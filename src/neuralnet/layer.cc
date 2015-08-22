#include "neuralnet/layer.h"

#include <glog/logging.h>
#include <algorithm>
#include "mshadow/tensor.h"
#include "mshadow/cxxnet_op.h"
#include "utils/singleton.h"

namespace singa {

using mshadow::cpu;
using mshadow::expr::broadcast;
using mshadow::expr::chpool;
using mshadow::expr::F;
using mshadow::expr::pool;
using mshadow::expr::sumall_except_dim;
using mshadow::expr::unpool;
using mshadow::op::power;
using mshadow::op::relu;
using mshadow::op::relu_grad;
using mshadow::op::sigmoid;
using mshadow::op::square;
using mshadow::op::stanh;
using mshadow::op::stanh_grad;
using mshadow::op::threshold;
using mshadow::Random;
using mshadow::red::maximum;
using mshadow::red::sum;
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

/***************Implementation for ShardDataLayer**************************/
ShardDataLayer::~ShardDataLayer() {
  if (shard_ != nullptr)
    delete shard_;
  shard_ = nullptr;
}

void ShardDataLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  shard_ = new DataShard(proto.sharddata_conf().path(), DataShard::kRead);
  string key;
  shard_->Next(&key, &sample_);
  delete shard_;
  shard_ = nullptr;
  batchsize_ = proto.sharddata_conf().batchsize();
  if (partition_dim() == 0)
    batchsize_ /= npartitions;
  records_.resize(batchsize_);
  random_skip_ = proto.sharddata_conf().random_skip();
}

void ShardDataLayer::ComputeFeature(int flag, Metric* perf) {
  if ((flag & kForward) == 0)
    return;

  if (shard_ == nullptr)
    shard_ = new DataShard(layer_proto_.sharddata_conf().path(),
                           DataShard::kRead);
  if (random_skip_) {
    int nskip = rand() % random_skip_;
    LOG(INFO) << "Random Skip " << nskip << " records, there are "
              << shard_->Count() << " records in total";
    string key;
    for (int i = 0; i < nskip; i++) {
      shard_->Next(&key, &sample_);
    }
    random_skip_ = 0;
  }
  for (auto& record : records_) {
    string key;
    if (!shard_->Next(&key, &record)) {
      shard_->SeekToFirst();
      CHECK(shard_->Next(&key, &record));
    }
  }
}

/********* Implementation for LabelLayer **************/
void LabelLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 1);
  int batchsize = static_cast<DataLayer*>(srclayers_[0])->batchsize();
  data_.Reshape(vector<int>{batchsize});
}

void LabelLayer::ParseRecords(int flag, const vector<Record>& records,
                              Blob<float>* blob) {
  int rid = 0;
  float *label = blob->mutable_cpu_data();
  for (const Record& record : records) {
    label[rid++] = record.image().label();
    // CHECK_LT(record.image().label(),10);
  }
  CHECK_EQ(rid, blob->shape()[0]);
}

/**************** Implementation for MnistLayer ******************/
void MnistLayer::ParseRecords(int flag,
    const vector<Record>& records, Blob<float>* blob){
  if ((flag & kForward) == 0)
    return;
  LOG_IF(ERROR, records.size()==0)<<"Empty records to parse";
  int ndim=records.at(0).image().shape_size();
  int inputsize =records.at(0).image().shape(ndim-1);
  CHECK_EQ(inputsize, blob->shape()[2]);

  float* dptr=blob->mutable_cpu_data();
  for(const Record& record: records){
    const SingleLabelImageRecord& imagerecord=record.image();
    if(imagerecord.pixel().size()) {
      string pixel=imagerecord.pixel();
      for(int i = 0, k = 0; i < inputsize; i++) {
        for(int j = 0; j < inputsize; j++) {
          // NOTE!!! must cast pixel to uint8_t then to float!!! waste a lot of
          // time to debug this
          float x =  static_cast<float>(static_cast<uint8_t>(pixel[k++]));
          x = x / norm_a_-norm_b_;
          *dptr = x;
          dptr++;
        }
      }
    } else {
      for(int i = 0, k = 0; i < inputsize; i++) {
        for(int j = 0; j < inputsize; j++) {
          *dptr = imagerecord.data(k++) / norm_a_ - norm_b_;
          dptr++;
        }
      }
    }
  }
  CHECK_EQ(dptr, blob->mutable_cpu_data()+blob->count());
}
void MnistLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 1);
  int batchsize = static_cast<DataLayer*>(srclayers_[0])->batchsize();
  Record sample = static_cast<DataLayer*>(srclayers_[0])->sample();
  kernel_ = proto.mnist_conf().kernel();
  sigma_ = proto.mnist_conf().sigma();
  alpha_ = proto.mnist_conf().alpha();
  beta_ = proto.mnist_conf().beta();
  gamma_ = proto.mnist_conf().gamma();
  resize_ = proto.mnist_conf().resize();
  norm_a_ = proto.mnist_conf().norm_a();
  norm_b_ = proto.mnist_conf().norm_b();
  elastic_freq_ = proto.mnist_conf().elastic_freq();
  int ndim = sample.image().shape_size();
  CHECK_GE(ndim, 2);
  if (resize_) {
    data_.Reshape(vector<int>{batchsize, 1, resize_, resize_});
  } else {
    int s = sample.image().shape(ndim - 1);
    CHECK_EQ(s, sample.image().shape(ndim - 2));
    data_.Reshape(vector<int>{batchsize, 1, s, s});
  }
}

/*************** Implementation for RGBImageLayer *************************/
void RGBImageLayer::ParseRecords(int flag,
    const vector<Record>& records, Blob<float>* blob){
  if ((flag & kForward) == 0)
    return;

  const vector<int>& s=blob->shape();
  auto images = Tensor4(&data_);
  const SingleLabelImageRecord& r=records.at(0).image();
  Tensor<cpu, 3> raw_image(Shape3(r.shape(0),r.shape(1),r.shape(2)));
  AllocSpace(raw_image);
  Tensor<cpu, 3> croped_image(nullptr, Shape3(s[1],s[2],s[3]));
  if(cropsize_)
    AllocSpace(croped_image);
    //CHECK(std::equal(croped_image.shape(), raw_image.shape());
  int rid=0;
  const float* meandptr=mean_.cpu_data();
  for(const Record& record: records){
    auto image=images[rid];
    bool do_crop = cropsize_ > 0 && ((flag & kTrain) == kTrain);
    bool do_mirror = mirror_ && rand() % 2 && ((flag & kTrain) == kTrain);
    float* dptr=nullptr;
    if(do_crop||do_mirror)
      dptr=raw_image.dptr;
    else
      dptr=image.dptr;
    if(record.image().pixel().size()){
      string pixel=record.image().pixel();
      for(size_t i=0;i<pixel.size();i++)
        dptr[i]=static_cast<float>(static_cast<uint8_t>(pixel[i]));
    }else {
      memcpy(dptr, record.image().data().data(),
          sizeof(float)*record.image().data_size());
    }
    for(int i=0;i<mean_.count();i++)
      dptr[i]-=meandptr[i];

    if(do_crop){
      int hoff=rand()%(r.shape(1)-cropsize_);
      int woff=rand()%(r.shape(2)-cropsize_);
      Shape<2> cropshape=Shape2(cropsize_, cropsize_);
      if(do_mirror){
        croped_image=crop(raw_image, cropshape, hoff, woff);
        image=mirror(croped_image);
      }else{
        image=crop(raw_image, cropshape, hoff, woff);
      }
    }else if(do_mirror){
      image=mirror(raw_image);
    }
    rid++;
  }
}

void RGBImageLayer::Setup(const LayerProto& proto, int npartitions) {
  ParserLayer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 1);
  scale_ = proto.rgbimage_conf().scale();
  cropsize_ = proto.rgbimage_conf().cropsize();
  mirror_ = proto.rgbimage_conf().mirror();
  int batchsize = static_cast<DataLayer*>(srclayers_[0])->batchsize();
  Record sample = static_cast<DataLayer*>(srclayers_[0])->sample();
  vector<int> shape;
  shape.push_back(batchsize);
  for (int x : sample.image().shape()) {
    shape.push_back(x);
  }
  CHECK_EQ(shape.size(), 4);
  if (cropsize_) {
    shape[2] = cropsize_;
    shape[3] = cropsize_;
  }
  data_.Reshape(shape);
  mean_.Reshape({shape[1], shape[2], shape[3]});
  if (proto.rgbimage_conf().has_meanfile()) {
    if (proto.rgbimage_conf().meanfile().find("binaryproto") != string::npos) {
      CaffeBlob mean;
      ReadProtoFromBinaryFile(proto.rgbimage_conf().meanfile().c_str(), &mean);
      CHECK_EQ(mean_.count(), mean.data_size());
      memcpy(mean_.mutable_cpu_data(), mean.data().data(),
             sizeof(float)*mean.data_size());
    } else {
      SingleLabelImageRecord mean;
      ReadProtoFromBinaryFile(proto.rgbimage_conf().meanfile().c_str(), &mean);
      CHECK_EQ(mean_.count(), mean.data_size());
      memcpy(mean_.mutable_cpu_data(), mean.data().data(),
             sizeof(float)*mean.data_size());
    }
  } else {
    memset(mean_.mutable_cpu_data(), 0, sizeof(float) * mean_.count());
  }
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

void ConvolutionLayer::ComputeFeature(int flag, Metric* perf){
  auto src = Tensor4(srclayers_[0]->mutable_data(this));
  auto data = Tensor3(&data_);
  auto col = Tensor2(&col_data_);
  auto weight = Tensor2(weight_->mutable_data());
  auto bias = Tensor1(bias_->mutable_data());
  for (int n = 0; n < batchsize_; n++) {
    if (pad_ > 0)
      col = unpack_patch2col(pad(src[n], pad_), kernel_, stride_);
    else
      col = unpack_patch2col(src[n], kernel_, stride_);
    data[n] = dot(weight, col);
  }
  data += broadcast<1>(bias, data.shape);
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
  gbias = sumall_except_dim<1>(grad);
  gweight = 0.0f;
  Shape<3> padshp(gsrc.shape.SubShape());
  padshp[0] += 2 * pad_;
  padshp[1] += 2 * pad_;
  Shape<2> imgshp = Shape2(height_, width_);
  for (int n = 0; n < batchsize_; n++) {
    if (pad_ > 0)
      col = unpack_patch2col(pad(src[n], pad_), kernel_, stride_);
    else
      col = unpack_patch2col(src[n], kernel_, stride_);
    gweight += dot(grad[n], col.T());
    if (gsrcblob != nullptr) {
      gcol = dot(weight.T(), grad[n]);
      gsrc[n] = crop(pack_col2patch(gcol, padshp, kernel_, stride_), imgshp);
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
  if((flag & kTrain) != kTrain) {
    data_.CopyFrom(srclayers_[0]->data(this));
    return;
  }
  float pkeep = 1 - pdrop_;
  auto mask = Tensor1(&mask_);
  mask = F<threshold>(TSingleton<Random<cpu>>::Instance() \
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
/**************** Implementation for RBMVisLayer********************/
RBMVisLayer::~RBMVisLayer() {
  delete weight_;
  delete bias_;
}

void RBMVisLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
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
  data_.ReshapeLike(src);  // this is visible dimension
  neg_data_.ReshapeLike(data_);
  neg_sample_.ReshapeLike(data_);
  weight_ = Param::Create(proto.param(0));
  bias_ = Param::Create(proto.param(1));
  bias_->Setup(vector<int>{src.count() / batchsize_});
}
Blob<float>* RBMVisLayer::Sample(int flag) {
  Tensor<cpu, 2> sample, data;
  if ((flag & kPositive) == kPositive) {
    LOG(FATAL) << "RBMVisLayer can not be sampled for positive flag";
  } else {
    data = Tensor2(&neg_data_);
    sample = Tensor2(&neg_sample_);
  }
  auto random = TSingleton<Random<cpu>>::Instance();
  random->SampleBinary(sample, data);
  return &neg_sample_;
}
void RBMVisLayer::ComputeFeature(int flag, Metric* perf) {
  if ((flag & kPositive) == kPositive) { /*positive flag*/
    data_.CopyFrom(input_layer_->data(this), true);
  } else if ((flag & kNegative) == kNegative) {   /*negative flag*/
    auto hid_sample = Tensor2(hid_layer_->Sample(flag));
    // fetch sampling results from hidden layer
    auto data = Tensor2(&neg_data_);
    auto weight = Tensor2(weight_->mutable_data());
    auto bias = Tensor1(bias_->mutable_data());
    data = dot(hid_sample, weight);
    data += repmat(bias, batchsize_);
    data = F<op::sigmoid>(data);
    if ((flag & kTest) == kTest) {
      const float *dptr = data_.cpu_data(), *rcns = neg_data_.cpu_data();
      float err = 0.f;
      for (int i = 0; i < data_.count(); i++) {
        err += (dptr[i] - rcns[i]) * (dptr[i] - rcns[i]);
      }
      perf->Add("Squared Error", err / batchsize_);
    }
  }
}

void RBMVisLayer::ComputeGradient(int flag, Metric* perf) {
  auto vis_pos = Tensor2(&data_);
  auto vis_neg = Tensor2(&neg_data_);
    auto gbias = Tensor1(bias_->mutable_grad());
  gbias = sum_rows(vis_neg);
  gbias -= sum_rows(vis_pos);
}
/**************** Implementation for RBMHidLayer********************/
RBMHidLayer::~RBMHidLayer() {
  delete weight_;
  delete bias_;
}

void RBMHidLayer::Setup(const LayerProto& proto,
      int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 1);
  const auto& src_data = srclayers_[0]->data(this);
  batchsize_ = src_data.shape()[0];
  vdim_ = src_data.count()/batchsize_;
  hdim_ = proto.rbmhid_conf().hid_dim();
  gaussian_ = proto.rbmhid_conf().gaussian();
  data_.Reshape(vector<int>{batchsize_, hdim_});
  neg_data_.ReshapeLike(data_);
  sample_.ReshapeLike(data_);
  neg_sample_.ReshapeLike(data_);
  weight_ = Param::Create(proto.param(0));
  bias_ = Param::Create(proto.param(1));
  bias_->Setup(vector<int>{hdim_});
  weight_->Setup(vector<int>{hdim_, vdim_});
  vis_layer_ = static_cast<RBMVisLayer*> (srclayers_[0]);
}

Blob<float>* RBMHidLayer::Sample(int flag) {
  Tensor<cpu, 2> sample, data;
  if ((flag & kPositive) == kPositive) {
    data = Tensor2(&data_);
    sample = Tensor2(&sample_);
  } else {
    data = Tensor2(&neg_data_);
    sample = Tensor2(&neg_sample_);
  }
  auto random = TSingleton<Random<cpu>>::Instance();
  if (gaussian_) {  // first gibbs
    random->SampleGaussian(sample, 0.0f, 1.0f);
    sample += data;
  } else {
    random->SampleBinary(sample, data);
  }
  return (flag & kPositive) == kPositive ? &sample_ : &neg_sample_;
}

void RBMHidLayer::ComputeFeature(int flag, Metric* perf) {
  auto weight = Tensor2(weight_->mutable_data());
  auto bias = Tensor1(bias_->mutable_data());

  Tensor<cpu, 2> data, src;
  if ((flag & kPositive) == kPositive) {  /*postive flag*/
    data = Tensor2(&data_);
    src = Tensor2(vis_layer_->mutable_data(this));
  } else {
    data = Tensor2(&neg_data_);
    src = Tensor2(vis_layer_->Sample(flag));
  }
  data = dot(src, weight.T());
  data += repmat(bias, batchsize_);

  if (!gaussian_)
    data = F<op::sigmoid>(data);
}

void RBMHidLayer::ComputeGradient(int flag, Metric* perf) {
  auto hid_pos = Tensor2(&data_);
  auto hid_neg = Tensor2(&neg_data_);
  auto vis_pos = Tensor2(vis_layer_->mutable_data(this));
  auto vis_neg = Tensor2(vis_layer_->mutable_data(this));

  auto gbias = Tensor1(bias_->mutable_grad());
  gbias = sum_rows(hid_neg);
  gbias -= sum_rows(hid_pos);
  gbias /= batchsize_;

  auto gweight = Tensor2(weight_->mutable_grad());
  gweight = dot(hid_neg.T(), vis_neg);
  gweight -= dot(hid_pos.T(), vis_pos);
  gweight /= batchsize_;
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
  data += repmat(bias, batchsize_);
}

void InnerProductLayer::ComputeGradient(int flag, Metric* perf) {
  if ((flag & kForward) != kForward)
    return;
  auto src = Tensor2(srclayers_[0]->mutable_data(this));
  auto grad = Tensor2(&grad_);
  auto weight = Tensor2(weight_->mutable_data());
  auto gweight = Tensor2(weight_->mutable_grad());
  auto gbias = Tensor1(bias_->mutable_grad());

  gbias = sum_rows(grad);
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
/*****************************************************************************
 * Implementation for LabelLayer
 *****************************************************************************/
void LabelLayer::Setup(const LayerProto& proto, int npartitions){
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(),1);
  int batchsize=static_cast<DataLayer*>(srclayers_[0])->batchsize();
  data_.Reshape(vector<int>{batchsize});
}

void LabelLayer::ParseRecords(int flag, const vector<Record>& records,
    Blob<float>* blob){
  int rid=0;
  float *label= blob->mutable_cpu_data() ;
  for(const Record& record: records){
    label[rid++]=record.image().label();
    //  CHECK_LT(record.image().label(),10);
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
  norm = chpool<sum>(F<square>(src), lsize_) * salpha + knorm_;
  data = src * F<power>(norm, -beta_);
}

void LRNLayer::ComputeGradient(int flag, Metric* perf) {
  const float salpha = alpha_ / lsize_;
  auto src = Tensor4(srclayers_[0]->mutable_data(this));
  auto norm = Tensor4(&norm_);
  auto grad = Tensor4(&grad_);
  auto gsrc = Tensor4(srclayers_[0]->mutable_grad(this));

  gsrc = grad * F<op::power>( norm, -beta_ );
  gsrc += ( - 2.0f * beta_ * salpha ) * chpool<red::sum>(
      grad * src * F<op::power>( norm, -beta_-1.0f ), lsize_ )  * src;
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
    data = pool<maximum>(src, kernel_, stride_);
  else if (pool_ == PoolingProto_PoolMethod_AVE)
    data = pool<sum>(src, kernel_, stride_) * (1.0f / (kernel_ * kernel_));
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
    gsrc = unpool<maximum>(src, data, grad, kernel_, stride_);
  else if (pool_ == PoolingProto_PoolMethod_AVE)
    gsrc = unpool<sum>(src, data, grad, kernel_, stride_)
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
  data = F<relu>(src);
}

void ReLULayer::ComputeGradient(int flag, Metric* perf) {
  auto data = Tensor1(&data_);
  auto grad = Tensor1(&grad_);
  auto gsrc = Tensor1(srclayers_[0]->mutable_grad(this));
  gsrc = F<relu_grad>(data)*grad;
}

/**************** Implementation for RBMHidLayer********************/
RBMHidLayer::~RBMHidLayer() {
  delete weight_;
  delete bias_;
}
void RBMHidLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 1);
  const auto& src_data = srclayers_[0]->data(this, kPositive);
  const auto& src_sample = srclayers_[0]->data(this, kNegative);
  scale_ = static_cast<float> (1.0f);
  batchsize_ = src_data.shape()[0];
  neg_batchsize_ = src_sample.shape()[0];
  vdim_ = src_data.count() / batchsize_;
  hdim_ = proto.rbmhid_conf().hid_dim();
  data_.Reshape(vector<int>{batchsize_, hdim_});
  hid_sample_.Reshape(vector<int>{neg_batchsize_, hdim_});
  weight_ = Param::Create(proto.param(0));
  bias_ = Param::Create(proto.param(1));
  weight_->Setup(proto.param(0), vector<int>{vdim_, hdim_});
  bias_->Setup(proto.param(1), vector<int>{hdim_});
}

void RBMHidLayer::ComputeGradient(int flag, Metric* perf) {
  auto data = Tensor2(&data_);
  auto hid_sample = Tensor2(&hid_sample_);
  auto gbias = Tensor1(bias_->mutable_grad());
  gbias = sum_rows(hid_sample);
  gbias -= sum_rows(data);
  gbias *= scale_ / (1.0f * batchsize_);
}

/**************** Implementation for RBMVisLayer********************/
RBMVisLayer::~RBMVisLayer() {
  delete weight_;
  delete bias_;
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
  data = F<op::sigmoid>(src);
}

void SigmoidLayer::ComputeGradient(int flag, Metric* perf) {
  auto data = Tensor1(&data_);
  auto grad = Tensor1(&grad_);
  auto gsrc = Tensor1(srclayers_[0]->mutable_grad(this));
  gsrc = F<op::sigmoid_grad>(data)*grad;
}
/*******************Implementation of TanLayer***************************/
void TanhLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  data_.ReshapeLike(srclayers_[0]->data(this));
  grad_.ReshapeLike(srclayers_[0]->grad(this));
}

void TanhLayer::ComputeFeature(int flag, Metric* perf) {
  auto data = Tensor1(&data_);
  auto src = Tensor1(srclayers_[0]->mutable_data(this));
  data = F<stanh>(src);
}

void TanhLayer::ComputeGradient(int flag, Metric* perf) {
  auto data = Tensor1(&data_);
  auto grad = Tensor1(&grad_);
  auto gsrc = Tensor1(srclayers_[0]->mutable_grad(this));
  gsrc = F<stanh_grad>(data) * grad;
}
/********** * Implementation for EuclideanLossLayer*************************/
void EuclideanLossLayer::Setup(const LayerProto& proto, int npartitions) {
  LossLayer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 2);
  data_.Reshape(srclayers_[0]->data(this).shape());
  batchsize_ = data_.shape()[0];
  dim_ = data_.count()/batchsize_;
  metric_.Reshape(vector<int>{1});
}
void EuclideanLossLayer::ComputeFeature(int flag, Metric* perf) {
  const float* reconstruct_dptr = srclayers_[0]->data(this).cpu_data();
  const float* input_dptr = srclayers_[1]->data(this).cpu_data();
  float loss = 0;
  for (int n = 0; n < batchsize_; n++) {
    for (int j = 0; j < dim_; ++j) {
      loss += (input_dptr[j] - reconstruct_dptr[j]) *
        (input_dptr[j] - reconstruct_dptr[j]);
    }
    reconstruct_dptr +=dim_;
    input_dptr +=dim_;
  }
  CHECK_EQ(reconstruct_dptr,
      srclayers_[0]->data(this).cpu_data() + (batchsize_*dim_));
  CHECK_EQ(input_dptr,
      srclayers_[1]->data(this).cpu_data() + (batchsize_*dim_));
  perf->Add("loss", loss / batchsize_);
}
void EuclideanLossLayer::ComputeGradient(int flag, Metric* perf) {
  const float* reconstruct_dptr = srclayers_[0]->data(this).cpu_data();
  const float* input_dptr = srclayers_[1]->data(this).cpu_data();
  Blob<float>* gsrcblob = srclayers_[0]->mutable_grad(this);
  float* gsrcptr = gsrcblob->mutable_cpu_data();
  for (int n = 0; n < batchsize_; n++) {
    for (int j = 0; j < dim_; j++)
    gsrcptr[n*dim_+j] = 2 * (reconstruct_dptr[n*dim_+j]-input_dptr[n*dim_+j]);
  }
  Tensor<cpu, 1> gsrc(gsrcptr, Shape1(gsrcblob->count()));
  gsrc /= batchsize_;
}

/********** * Implementation for SoftmaxLossLayer*************************/
void SoftmaxLossLayer::Setup(const LayerProto& proto, int npartitions) {
  LossLayer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 2);
  data_.Reshape(srclayers_[0]->data(this).shape());
  batchsize_ = data_.shape()[0];
  dim_ = data_.count() / batchsize_;
  topk_ = proto.softmaxloss_conf().topk();
  metric_.Reshape(vector<int>{2});
  scale_ = proto.softmaxloss_conf().scale();
}
void SoftmaxLossLayer::ComputeFeature(int flag, Metric* perf) {
  Shape<2> s=Shape2(batchsize_, dim_);
  Tensor<cpu, 2> prob(data_.mutable_cpu_data(), s);
  Tensor<cpu, 2> src(srclayers_[0]->mutable_data(this)->mutable_cpu_data(), s);
  Softmax(prob, src);
  const float* label = srclayers_[1]->data(this).cpu_data();
  const float* probptr = prob.dptr;
  float loss = 0, precision = 0;
  for (int n = 0; n < batchsize_; n++) {
    int ilabel = static_cast<int>(label[n]);
    //  CHECK_LT(ilabel,10);
    CHECK_GE(ilabel, 0);
    float prob_of_truth = probptr[ilabel];
    loss -= log(std::max(prob_of_truth, FLT_MIN));
    vector<std::pair<float, int> > probvec;
    for (int j = 0; j < dim_; ++j) {
      probvec.push_back(std::make_pair(probptr[j], j));
    }
    std::partial_sort(probvec.begin(), probvec.begin() + topk_, probvec.end(),
                      std::greater<std::pair<float, int> >());
    // check if true label is in top k predictions
    for (int k = 0; k < topk_; k++) {
      if (probvec[k].second == static_cast<int>(label[n])) {
        precision++;
        break;
      }
    }
    probptr += dim_;
  }
  CHECK_EQ(probptr, prob.dptr+prob.shape.Size());
  perf->Add("loss", loss * scale_ / (1.0f * batchsize_));
  perf->Add("accuracy", precision * scale_ / (1.0f * batchsize_));
}

void SoftmaxLossLayer::ComputeGradient(int flag, Metric* perf) {
  const float* label = srclayers_[1]->data(this).cpu_data();
  Blob<float>* gsrcblob = srclayers_[0]->mutable_grad(this);
  gsrcblob->CopyFrom(data_);
  float* gsrcptr = gsrcblob->mutable_cpu_data();
  for (int n = 0; n < batchsize_; n++) {
    gsrcptr[n*dim_ + static_cast<int>(label[n])] -= 1.0f;
  }
  Tensor<cpu, 1> gsrc(gsrcptr, Shape1(gsrcblob->count()));
  gsrc *= scale_ / (1.0f * batchsize_);
}

/********* Implementation for BridgeDstLayer **************/
void BridgeDstLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 1);
  data_.Reshape(srclayers_[0]->data(this).shape());
  grad_.ReshapeLike(data_);
}

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
}

void SliceLayer::ComputeFeature(int flag, Metric *perf) {
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
}

void SliceLayer::ComputeGradient(int flag, Metric* perf) {
  // LOG(FATAL) << "Not implemented";
}

int SliceLayer::SliceID(const Layer* layer) const {
  CHECK(layer != nullptr);
  for (size_t i = 0; i < datavec_.size(); i++) {
    // LOG(ERROR)<<"get slice "<<IntVecToString(shapes_[i]);
    if (dstlayers_[i] == layer)
      return i;
  }
  CHECK(false);
  return -1;
}

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
