#include <glog/logging.h>
#include <memory>
#include <algorithm>
#include "mshadow/tensor.h"
#include "mshadow/cxxnet_op.h"
#include "neuralnet/layer.h"
#include "utils/singleton.h"
#include "utils/factory.h"

using namespace mshadow;
using namespace mshadow::expr;

namespace singa {
inline Tensor<cpu, 4> Tensor4(Blob<float>* blob) {
  const vector<int>& shape = blob->shape();
  Tensor<cpu, 4> tensor(blob->mutable_cpu_data(),
      Shape4(shape[0], shape[1], shape[2], shape[3]));
  return tensor;
}

inline Tensor<cpu, 3> Tensor3(Blob<float>* blob){
  const vector<int>& shape = blob->shape();
  Tensor<cpu, 3> tensor(blob->mutable_cpu_data(),
      Shape3(shape[0], shape[1], blob->count() / shape[0] / shape[1]));
  return tensor;
}
inline Tensor<cpu, 2> Tensor2(Blob<float>* blob){
  const vector<int>& shape = blob->shape();
  Tensor<cpu, 2> tensor(blob->mutable_cpu_data(),
      Shape2(shape[0], blob->count() / shape[0]));
  return tensor;
}
inline Tensor<cpu, 1> Tensor1(Blob<float>* blob){
  Tensor<cpu, 1> tensor(blob->mutable_cpu_data(), Shape1(blob->count()));
  return tensor;
}

/************ Implementation for ConvProductLayer*************************/
ConvolutionLayer::~ConvolutionLayer() {
  delete weight_;
  delete bias_;
}
void ConvolutionLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  ConvolutionProto conv_conf=proto.convolution_conf();
  kernel_=conv_conf.kernel();
  CHECK_GT(kernel_, 0) << "Filter size cannot be zero.";
  pad_=conv_conf.pad();
  stride_=conv_conf.stride();
  num_filters_=conv_conf.num_filters();
  if(partition_dim() > 0)
    num_filters_ /= npartitions;

  const vector<int>& srcshape=srclayers_[0]->data(this).shape();
  int dim=srcshape.size();
  CHECK_GT(dim, 2);
  width_=srcshape[dim-1];
  height_=srcshape[dim-2];
  if(dim>3)
    channels_=srcshape[dim-3];
  else if(dim>2)
    channels_=1;
  batchsize_=srcshape[0];
  conv_height_=(height_ + 2 * pad_ - kernel_) / stride_ + 1;
  conv_width_= (width_ + 2 * pad_ - kernel_) / stride_ + 1;
  col_height_=channels_*kernel_*kernel_;
  col_width_=conv_height_*conv_width_;
  vector<int> shape{batchsize_, num_filters_, conv_height_, conv_width_};
  data_.Reshape(shape);
  grad_.Reshape(shape);
  col_data_.Reshape(vector<int>{col_height_, col_width_});
  col_grad_.Reshape(vector<int>{col_height_, col_width_});

  Factory<Param>* factory=Singleton<Factory<Param>>::Instance();
  weight_ = factory->Create("Param");
  weight_->Setup(proto.param(0), vector<int>{num_filters_, col_height_});
  bias_ = factory->Create("Param");
  bias_->Setup(proto.param(1), vector<int>{num_filters_});
}

void ConvolutionLayer::ComputeFeature(Phase phase, Metric* perf){
  auto src = Tensor4(srclayers_[0]->mutable_data(this));
  auto data = Tensor3(&data_);
  auto col = Tensor2(&col_data_);
  auto weight = Tensor2(weight_->mutable_data());
  auto bias = Tensor1(bias_->mutable_data());

  for(int n=0;n<batchsize_;n++){
    if(pad_>0)
      col=unpack_patch2col(pad(src[n], pad_), kernel_, stride_);
    else
      col=unpack_patch2col(src[n], kernel_, stride_);
    data[n]=dot(weight, col);
  }
  data+=broadcast<1>(bias, data.shape);
}

void ConvolutionLayer::ComputeGradient(Phase phase) {
  auto src = Tensor4(srclayers_[0]->mutable_data(this));
  auto col = Tensor2(&col_data_);
  auto weight = Tensor2(weight_->mutable_data());

  auto grad = Tensor3(&grad_);
  auto gcol = Tensor2(&col_grad_);
  auto gweight = Tensor2(weight_->mutable_grad());
  auto gbias = Tensor1(bias_->mutable_grad());

  Blob<float>* gsrcblob=srclayers_[0]->mutable_grad(this);
  Tensor<cpu, 4> gsrc(nullptr, Shape4(batchsize_, channels_, height_, width_));
  if(gsrcblob!=nullptr)
    gsrc.dptr=gsrcblob->mutable_cpu_data();
  gbias=sumall_except_dim<1>(grad);

  gweight = 0.0f;
  Shape<3> padshp(gsrc.shape.SubShape());
  padshp[0] += 2 * pad_;
  padshp[1] += 2 * pad_;
  Shape<2> imgshp = Shape2(height_, width_);
  for(int n=0;n<batchsize_;n++){
    if(pad_>0)
      col=unpack_patch2col(pad(src[n], pad_), kernel_, stride_);
    else
      col=unpack_patch2col(src[n], kernel_, stride_);
    gweight += dot(grad[n], col.T());

    if(gsrcblob!=nullptr){
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

void DropoutLayer::ComputeFeature(Phase phase, Metric* perf) {
  // check training
  if(phase != kTrain){//!training){
    data_.CopyFrom(srclayers_[0]->data(this));
    return;
  }
  float pkeep=1-pdrop_;
  auto mask = Tensor1(&mask_);
  mask = F<op::threshold>(TSingleton<Random<cpu>>::Instance()\
      ->uniform(mask.shape), pkeep ) * (1.0f/pkeep);
  auto data = Tensor1(&data_);
  auto src = Tensor1(srclayers_[0]->mutable_data(this));
  data = src * mask;
}

void DropoutLayer::ComputeGradient(Phase phase)  {
  auto mask = Tensor1(&mask_);
  auto grad = Tensor1(&grad_);
  auto gsrc = Tensor1(srclayers_[0]->mutable_grad(this));
  gsrc = grad * mask;
}

/*********** Implementation for InnerProductLayer**********/
InnerProductLayer::~InnerProductLayer() {
  delete weight_;
  delete bias_;
}
void InnerProductLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(), 1);
  const auto& src=srclayers_[0]->data(this);
  batchsize_=src.shape()[0];
  vdim_=src.count()/batchsize_;
  hdim_=proto.innerproduct_conf().num_output();
  if(partition_dim()>0)
    hdim_ /= npartitions;
  data_.Reshape(vector<int>{batchsize_, hdim_});
  grad_.ReshapeLike(data_);
  Factory<Param>* factory=Singleton<Factory<Param>>::Instance();
  weight_ = factory->Create("Param");
  bias_ = factory->Create("Param");
  weight_->Setup(proto.param(0), vector<int>{vdim_, hdim_});
  bias_->Setup(proto.param(1), vector<int>{hdim_});
}

void InnerProductLayer::ComputeFeature(Phase phase, Metric* perf) {
  auto data = Tensor2(&data_);
  auto src = Tensor2(srclayers_[0]->mutable_data(this));
  auto weight = Tensor2(weight_->mutable_data());
  auto bias = Tensor1(bias_->mutable_data());
  data=dot(src, weight);
  // repmat: repeat bias vector into batchsize rows
  data+=repmat(bias, batchsize_);
}

void InnerProductLayer::ComputeGradient(Phase phas) {
  auto src = Tensor2(srclayers_[0]->mutable_data(this));
  auto grad = Tensor2(&grad_);
  auto weight = Tensor2(weight_->mutable_data());
  auto gweight = Tensor2(weight_->mutable_grad());
  auto gbias = Tensor1(bias_->mutable_grad());

  gbias=sum_rows(grad);
  gweight=dot(src.T(), grad);
  if(srclayers_[0]->mutable_grad(this)!=nullptr){
    auto gsrc = Tensor2(srclayers_[0]->mutable_grad(this));
    gsrc=dot(grad, weight.T());
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

void LabelLayer::ParseRecords(Phase phase, const vector<Record>& records,
    Blob<float>* blob){
  int rid=0;
  float *label= blob->mutable_cpu_data() ;
  for(const Record& record: records){
    label[rid++]=record.image().label();
    CHECK_LT(record.image().label(),10);
  }
  CHECK_EQ(rid, blob->shape()[0]);
}

/***************** Implementation for LRNLayer *************************/
void LRNLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(),1);
  lsize_ = proto.lrn_conf().local_size();
  CHECK_EQ(lsize_ % 2, 1) << "LRN only supports odd values for Localvol";
  knorm_=proto.lrn_conf().knorm();
  alpha_ = proto.lrn_conf().alpha();
  beta_ = proto.lrn_conf().beta();

  const vector<int>& s=srclayers_[0]->data(this).shape();
  data_.Reshape(s);
  grad_.Reshape(s);
  norm_.Reshape(s);
  batchsize_=s[0];
  channels_=s[1];
  height_=s[2];
  width_=s[3];
}

void LRNLayer::ComputeFeature(Phase phase, Metric* perf) {
  const float salpha = alpha_ / lsize_;
  auto src = Tensor4(srclayers_[0]->mutable_data(this));
  auto data = Tensor4(&data_);
  auto norm = Tensor4(&norm_);
  // stores normalizer without power
  norm= chpool<red::sum>( F<op::square>(src) , lsize_ ) * salpha + knorm_;
  data = src * F<op::power>(norm, -beta_ );
}

void LRNLayer::ComputeGradient(Phase phase) {
  const float salpha = alpha_ / lsize_;
  auto src = Tensor4(srclayers_[0]->mutable_data(this));
  auto norm = Tensor4(&norm_);
  auto grad = Tensor4(&grad_);
  auto gsrc = Tensor4(srclayers_[0]->mutable_grad(this));

  gsrc = grad * F<op::power>( norm, -beta_ );
  gsrc += ( - 2.0f * beta_ * salpha ) * chpool<red::sum>(
      grad * src * F<op::power>( norm, -beta_-1.0f ), lsize_ )  * src;
}

/**************** Implementation for MnistImageLayer******************/

void MnistLayer::ParseRecords(Phase phase,
    const vector<Record>& records, Blob<float>* blob){
  LOG_IF(ERROR, records.size()==0)<<"Empty records to parse";
  int ndim=records.at(0).image().shape_size();
  int inputsize =records.at(0).image().shape(ndim-1);
  CHECK_EQ(inputsize, blob->shape()[1]);

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
  CHECK_EQ(srclayers_.size(),1);
  int batchsize=static_cast<DataLayer*>(srclayers_[0])->batchsize();
  Record sample=static_cast<DataLayer*>(srclayers_[0])->sample();
  kernel_=proto.mnist_conf().kernel();
  sigma_=proto.mnist_conf().sigma();
  alpha_=proto.mnist_conf().alpha();
  beta_=proto.mnist_conf().beta();
  gamma_=proto.mnist_conf().gamma();
  resize_=proto.mnist_conf().resize();
  norm_a_=proto.mnist_conf().norm_a();
  norm_b_=proto.mnist_conf().norm_b();
  elastic_freq_=proto.mnist_conf().elastic_freq();

  int ndim=sample.image().shape_size();
  CHECK_GE(ndim,2);
  if(resize_)
    data_.Reshape(vector<int>{batchsize, resize_, resize_});
  else{
    int s=sample.image().shape(ndim-1);
    CHECK_EQ(s,sample.image().shape(ndim-2));
    data_.Reshape(vector<int>{batchsize, s, s });
  }
}

/******************** Implementation for PoolingLayer******************/
void PoolingLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(),1);
  PoolingProto pool_conf = proto.pooling_conf();
  kernel_=pool_conf.kernel();
  stride_=pool_conf.stride();
  CHECK_LT(pad_, kernel_);
  pool_=proto.pooling_conf().pool();
  CHECK(pool_ == PoolingProto_PoolMethod_AVE
        || pool_ == PoolingProto_PoolMethod_MAX)
      << "Padding implemented only for average and max pooling.";

  const auto& srcshape=srclayers_[0]->data(this).shape();
  int dim=srcshape.size();
  CHECK_GT(dim,2);
  width_ = srcshape[dim-1];
  height_ = srcshape[dim-2];
  if(dim>3)
    channels_ = srcshape[dim-3];
  else
    channels_=1;
  batchsize_=srcshape[0];
  pooled_height_ = static_cast<int>((height_ - kernel_) / stride_) + 1;
  pooled_width_ = static_cast<int>(( width_ - kernel_) / stride_) + 1;
  data_.Reshape(vector<int>{batchsize_, channels_, pooled_height_, pooled_width_});
  grad_.ReshapeLike(data_);
}

void PoolingLayer::ComputeFeature(Phase phase, Metric* perf) {
  auto src = Tensor4(srclayers_[0]->mutable_data(this));
  auto data = Tensor4(&data_);
  if(pool_ == PoolingProto_PoolMethod_MAX)
    data=pool<red::maximum>(src, kernel_, stride_);
  else if(pool_ == PoolingProto_PoolMethod_AVE)
    data=pool<red::sum>(src, kernel_, stride_) *(1.0f/(kernel_*kernel_));
}

/*
 * partition only on num/channel dim
 * assume grad and data have the same paritition
 */
void PoolingLayer::ComputeGradient(Phase phase) {
  auto src = Tensor4(srclayers_[0]->mutable_data(this));
  auto gsrc = Tensor4(srclayers_[0]->mutable_grad(this));
  auto data = Tensor4(&data_);
  auto grad = Tensor4(&grad_);
  if(pool_ == PoolingProto_PoolMethod_MAX)
    gsrc = unpool<red::maximum>(src, data, grad, kernel_, stride_);
  else if(pool_ == PoolingProto_PoolMethod_AVE)
    gsrc = unpool<red::sum>(src, data, grad, kernel_, stride_)
      *(1.0f/(kernel_*kernel_));
}

/***************** Implementation for ReLULayer *****************************/

void ReLULayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  data_.ReshapeLike(srclayers_[0]->data(this));
  grad_.ReshapeLike(*(srclayers_[0]->mutable_grad(this)));
}

void ReLULayer::ComputeFeature(Phase phase, Metric* perf) {
  auto data = Tensor1(&data_);
  auto src = Tensor1(srclayers_[0]->mutable_data(this));
  data=F<op::relu>(src);
}

void ReLULayer::ComputeGradient(Phase phase) {
  auto data = Tensor1(&data_);
  auto grad = Tensor1(&grad_);
  auto gsrc = Tensor1(srclayers_[0]->mutable_grad(this));
  gsrc=F<op::relu_grad>(data)*grad;
}

/*************** Implementation for RGBImageLayer *************************/

void RGBImageLayer::ParseRecords(Phase phase,
    const vector<Record>& records, Blob<float>* blob){
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
    bool do_crop=cropsize_>0&&(phase == kTrain);
    bool do_mirror=mirror_&&rand()%2&&(phase == kTrain);
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
  if(scale_)
    images=images*scale_;

  FreeSpace(raw_image);
  if(cropsize_)
    FreeSpace(croped_image);
}
void RGBImageLayer::Setup(const LayerProto& proto, int npartitions) {
  ParserLayer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(),1);
  scale_=proto.rgbimage_conf().scale();
  cropsize_=proto.rgbimage_conf().cropsize();
  mirror_=proto.rgbimage_conf().mirror();
  int batchsize=static_cast<DataLayer*>(srclayers_[0])->batchsize();
  Record sample=static_cast<DataLayer*>(srclayers_[0])->sample();
  vector<int> shape;
  shape.push_back(batchsize);
  for(int x: sample.image().shape()){
    shape.push_back(x);
  }
  CHECK_EQ(shape.size(),4);
  if(cropsize_){
    shape[2]=cropsize_;
    shape[3]=cropsize_;
  }
  data_.Reshape(shape);
  mean_.Reshape({shape[1],shape[2],shape[3]});
  if(proto.rgbimage_conf().has_meanfile()){
    if(proto.rgbimage_conf().meanfile().find("binaryproto") != string::npos) {
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
    memset(mean_.mutable_cpu_data(),0,sizeof(float)*mean_.count());
  }
}

/***************Implementation for ShardDataLayer**************************/
void ShardDataLayer::ComputeFeature(Phase phase, Metric* perf){
  if(random_skip_){
    int nskip = rand() % random_skip_;
    LOG(INFO)<<"Random Skip "<<nskip<<" records, there are "<<shard_->Count()
      <<" records in total";
    string key;
    for(int i=0;i<nskip;i++){
      shard_->Next(&key, &sample_);
    }
    random_skip_=0;
  }
  for(auto& record: records_){
    string key;
    if(!shard_->Next(&key, &record)){
      shard_->SeekToFirst();
      CHECK(shard_->Next(&key, &record));
    }
  }
}

void ShardDataLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  shard_= std::make_shared<DataShard>(proto.sharddata_conf().path(),
      DataShard::kRead);
  string key;
  shard_->Next(&key, &sample_);
  batchsize_=proto.sharddata_conf().batchsize();
  if(partition_dim() == 0)
    batchsize_ /= npartitions;

  records_.resize(batchsize_);
  random_skip_=proto.sharddata_conf().random_skip();
}
/*******************Implementation of TanLayer***************************/
void TanhLayer::Setup(const LayerProto& proto, int npartitions){
  Layer::Setup(proto, npartitions);
  data_.ReshapeLike(srclayers_[0]->data(this));
  grad_.ReshapeLike(srclayers_[0]->grad(this));
}

void TanhLayer::ComputeFeature(Phase phase, Metric* perf) {
  auto data = Tensor1(&data_);
  auto src = Tensor1(srclayers_[0]->mutable_data(this));
  data=F<op::stanh>(src);
}

void TanhLayer::ComputeGradient(Phase phase) {
  auto data = Tensor1(&data_);
  auto grad = Tensor1(&grad_);
  auto gsrc = Tensor1(srclayers_[0]->mutable_grad(this));
  gsrc=F<op::stanh_grad>(data)*grad;
}
/********** * Implementation for SoftmaxLossLayer*************************/
void SoftmaxLossLayer::Setup(const LayerProto& proto, int npartitions) {
  LossLayer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(),2);
  data_.Reshape(srclayers_[0]->data(this).shape());
  batchsize_=data_.shape()[0];
  dim_=data_.count()/batchsize_;
  topk_=proto.softmaxloss_conf().topk();
  metric_.Reshape(vector<int>{2});
  scale_=proto.softmaxloss_conf().scale();
}
void SoftmaxLossLayer::ComputeFeature(Phase phase, Metric* perf) {
  Shape<2> s=Shape2(batchsize_, dim_);
  Tensor<cpu, 2> prob(data_.mutable_cpu_data(), s);
  Tensor<cpu, 2> src(srclayers_[0]->mutable_data(this)->mutable_cpu_data(), s);
  Softmax(prob, src);
  const float* label=srclayers_[1]->data(this).cpu_data();
  const float* probptr=prob.dptr;
  float loss=0, precision=0;
  for(int n=0;n<batchsize_;n++){
    int ilabel=static_cast<int>(label[n]);
    CHECK_LT(ilabel,10);
    CHECK_GE(ilabel,0);
    float prob_of_truth=probptr[ilabel];
    loss-=log(std::max(prob_of_truth, FLT_MIN));
    vector<std::pair<float, int> > probvec;
    for (int j = 0; j < dim_; ++j) {
      probvec.push_back(std::make_pair(probptr[j], j));
    }
    std::partial_sort(
        probvec.begin(), probvec.begin() + topk_,
        probvec.end(), std::greater<std::pair<float, int> >());
    // check if true label is in top k predictions
    for (int k = 0; k < topk_; k++) {
      if (probvec[k].second == static_cast<int>(label[n])) {
        precision++;
        break;
      }
    }
    probptr+=dim_;
  }
  CHECK_EQ(probptr, prob.dptr+prob.shape.Size());
  perf->Add("loss", loss*scale_/(1.0f*batchsize_));
  perf->Add("accuracy", precision*scale_/(1.0f*batchsize_));
}

void SoftmaxLossLayer::ComputeGradient(Phase phase) {
  const float* label=srclayers_[1]->data(this).cpu_data();
  Blob<float>* gsrcblob=srclayers_[0]->mutable_grad(this);
  gsrcblob->CopyFrom(data_);
  float* gsrcptr=gsrcblob->mutable_cpu_data();
  for(int n=0;n<batchsize_;n++){
    gsrcptr[n*dim_+static_cast<int>(label[n])]-=1.0f;
  }
  Tensor<cpu, 1> gsrc(gsrcptr, Shape1(gsrcblob->count()));
  gsrc*=scale_/(1.0f*batchsize_);
}

}  // namespace singa
