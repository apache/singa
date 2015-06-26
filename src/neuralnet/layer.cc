#include <glog/logging.h>
#include <memory>
#include <algorithm>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "mshadow/tensor.h"
#include "mshadow/cxxnet_op.h"
#include "neuralnet/layer.h"
#include "utils/singleton.h"
#include "utils/factory.h"

using namespace mshadow;
using namespace mshadow::expr;

namespace singa {

/************ Implementation for ConvProductLayer*************************/
void ConvolutionLayer::Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  ConvolutionProto conv_conf=proto.convolution_conf();
  kernel_=conv_conf.kernel();
  CHECK_GT(kernel_, 0) << "Filter size cannot be zero.";
  pad_=conv_conf.pad();
  stride_=conv_conf.stride();
  num_filters_=conv_conf.num_filters();
  const vector<int>& srcshape=srclayers[0]->data(this).shape();
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
  weight_=shared_ptr<Param>(factory->Create("Param"));
  weight_->Setup(proto.param(0), vector<int>{num_filters_, col_height_});
  bias_=shared_ptr<Param>(factory->Create("Param"));
  bias_->Setup(proto.param(1), vector<int>{num_filters_});
}

void ConvolutionLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  LayerProto newproto(proto);
  ConvolutionProto *conv_conf=newproto.mutable_convolution_conf();
  conv_conf->set_num_filters(shape[1]);
  Setup(newproto, srclayers);
}

void ConvolutionLayer::ComputeFeature(Phase phase, const vector<SLayer>& srclayers){
  Tensor<cpu, 4> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(),
      Shape4(batchsize_, channels_, height_, width_));
  Tensor<cpu, 3> data(data_.mutable_cpu_data(),
      Shape3(batchsize_, num_filters_, conv_height_* conv_width_));
  Tensor<cpu, 2> col(col_data_.mutable_cpu_data(),
      Shape2(col_height_, col_width_));
  Tensor<cpu, 2> weight(weight_->mutable_cpu_data(),
      Shape2(num_filters_, col_height_));
  Tensor<cpu, 1> bias(bias_->mutable_cpu_data(),
      Shape1(num_filters_));

  for(int n=0;n<batchsize_;n++){
    if(pad_>0)
      col=unpack_patch2col(pad(src[n], pad_), kernel_, stride_);
    else
      col=unpack_patch2col(src[n], kernel_, stride_);
    data[n]=dot(weight, col);
  }
  data+=broadcast<1>(bias, data.shape);
}

void ConvolutionLayer::ComputeGradient(const vector<SLayer>& srclayers) {
  Tensor<cpu, 4> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(),
      Shape4(batchsize_, channels_, height_, width_));
  Tensor<cpu, 2> col(col_data_.mutable_cpu_data(),
      Shape2(col_height_, col_width_));
  Tensor<cpu, 2> weight(weight_->mutable_cpu_data(),
      Shape2(num_filters_, col_height_));

  Blob<float>* gsrcblob=srclayers[0]->mutable_grad(this);
  Tensor<cpu, 4> gsrc(nullptr, Shape4(batchsize_, channels_, height_, width_));
  if(gsrcblob!=nullptr)
    gsrc.dptr=gsrcblob->mutable_cpu_data();
  Tensor<cpu, 3> grad(grad_.mutable_cpu_data(),
      Shape3(batchsize_, num_filters_, conv_height_* conv_width_));
  Tensor<cpu, 2> gcol(col_grad_.mutable_cpu_data(),
      Shape2(col_height_, col_width_));
  Tensor<cpu, 2> gweight(weight_->mutable_cpu_grad(),
      Shape2(num_filters_, col_height_));
  Tensor<cpu, 1> gbias(bias_->mutable_cpu_grad(),
      Shape1(num_filters_));

  gweight=0.0f;
  gbias=sumall_except_dim<1>(grad);
  Shape<3> padshape(gsrc.shape.SubShape());
  padshape[0]+=2*pad_;padshape[1]+=2*pad_;
  Shape<2> imgshape=Shape2(height_, width_);
  for(int n=0;n<batchsize_;n++){
    if(pad_>0)
      col=unpack_patch2col(pad(src[n], pad_), kernel_, stride_);
    else
      col=unpack_patch2col(src[n], kernel_, stride_);
    gweight+=dot(grad[n], col.T());

    if(gsrcblob!=nullptr){
      gcol=dot(weight.T(), grad[n]);
      gsrc[n]=crop(pack_col2patch(gcol, padshape, kernel_, stride_), imgshape);
    }
  }
}

/****************** Implementation for DropoutLayer ***********************/
void DropoutLayer::Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers){
  data_.ReshapeLike(srclayers[0]->data(this));
  grad_.ReshapeLike(*srclayers[0]->mutable_grad(this));
  mask_.Reshape(srclayers[0]->data(this).shape());
  pdrop_=proto.dropout_conf().dropout_ratio();
}

void DropoutLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  Setup(proto, srclayers);
}

void DropoutLayer::ComputeFeature(Phase phase, const vector<SLayer>& srclayers) {
  // check training
  if(phase!= kTrain){//!training){
    data_.CopyFrom(srclayers[0]->data(this));
    return;
  }
  float pkeep=1-pdrop_;
  Tensor<cpu, 1> mask(mask_.mutable_cpu_data(), Shape1(mask_.count()));
  mask = F<op::threshold>(TSingleton<Random<cpu>>::Instance()\
      ->uniform(mask.shape), pkeep ) * (1.0f/pkeep);
  Tensor<cpu, 1> data(data_.mutable_cpu_data(), Shape1(data_.count()));
  Blob<float>* srcblob=srclayers[0]->mutable_data(this);
  Tensor<cpu, 1> src(srcblob->mutable_cpu_data(), Shape1(srcblob->count()));
  data=src*mask;
}

void DropoutLayer::ComputeGradient(const vector<SLayer>& srclayers)  {
  Tensor<cpu, 1> grad(grad_.mutable_cpu_data(), Shape1(data_.count()));
  Tensor<cpu, 1> mask(mask_.mutable_cpu_data(), Shape1(mask_.count()));
  Blob<float>* gsrcblob=srclayers[0]->mutable_grad(this);
  Tensor<cpu, 1> gsrc(gsrcblob->mutable_cpu_data(), Shape1(gsrcblob->count()));
  gsrc=grad*mask;
}
/**************** Implementation for DBMBottomLayer********************/
void DBMBottomLayer::Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers) {
  CHECK_EQ(srclayers.size(), 1);
  kPhase = true;
  const auto& possrc = srclayers[0]->data(this);
  is_first_iteration_bottom = true;
  batchsize_ = possrc.shape()[0];
  neg_batchsize_ = batchsize_;
  /*gibbs sampling size and input have the same size*/
  vdim_ = possrc.count()/batchsize_;
  hdim_ = proto.dbmbottom_conf().num_output();
  data_.Reshape(vector<int>{batchsize_, hdim_});
  hidden_data_.Reshape(vector<int>{neg_batchsize_, hdim_});
  scale_ = static_cast<float> (1.0f);
  negsrc_.Reshape(vector<int>{neg_batchsize_, vdim_});
  Factory<Param>* factory = Singleton<Factory<Param>>::Instance();
  weight_ = shared_ptr<Param>(factory->Create("Param"));
  bias_ = shared_ptr<Param>(factory->Create("Param"));
  weight_->Setup(proto.param(0), vector<int>{vdim_, hdim_}, vdim_*hdim_);
  bias_->Setup(proto.param(1), vector<int>{vdim_}, 0);
  srand((unsigned)time(NULL));
}
void DBMBottomLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers) {
}

void DBMBottomLayer::ComputeFeature(Phase phase, const vector<SLayer>& srclayers) {
  float matrix_norm = (weight_->data()).sum_data();
  float bias_norm = (bias_->data()).sum_data();
  if (phase == kPositive) { /*positive phase*/
        Tensor<cpu, 2> data(data_.mutable_cpu_data(),
          Shape2(batchsize_, hdim_));
        kPhase = true;
        CHECK_EQ(srclayers[0]->data(this).count(), batchsize_*vdim_);
        Tensor<cpu, 2> possrc(srclayers[0]->mutable_data(this)->mutable_cpu_data(),
         Shape2(batchsize_, vdim_));
        Tensor<cpu, 2> weight(weight_->mutable_cpu_data(),
          Shape2(vdim_, hdim_));
        data = dot(possrc, weight);
  }
  else if (phase == kNegative) {   /*negative phase*/
        if (is_first_iteration_bottom) {
          Tensor<cpu, 2> hidden_data(hidden_data_.mutable_cpu_data(),
            Shape2(neg_batchsize_, hdim_));
          Tensor<cpu, 2> weight(weight_->mutable_cpu_data(),
            Shape2(vdim_, hdim_));
          kPhase = true;
          CHECK_EQ(srclayers[0]->data(this).count(), batchsize_*vdim_);
          Tensor<cpu, 2> possrc(srclayers[0]->mutable_data(this)->mutable_cpu_data(),
          Shape2(batchsize_, vdim_));
          Tensor<cpu, 2> negsrc(negsrc_.mutable_cpu_data(),
            Shape2(neg_batchsize_, vdim_));
          for (int i = 0; i < batchsize_; i++)
            for (int j = 0; j < vdim_; j++)
              negsrc[i][j] = possrc[i][j];
            /*Use the input samples for initializing the gibbs chain*/
          hidden_data = dot(negsrc, weight);
          is_first_iteration_bottom = false;
        }
        else {
            Tensor<cpu, 2> hidden_data(hidden_data_.mutable_cpu_data(),
                           Shape2(neg_batchsize_, hdim_));
            Tensor<cpu, 2> negsrc(negsrc_.mutable_cpu_data(),
                           Shape2(neg_batchsize_, vdim_));
            Tensor<cpu, 2> weight(weight_->mutable_cpu_data(),
                           Shape2(vdim_, hdim_));
            Tensor<cpu, 1> bias(bias_->mutable_cpu_data(), Shape1(vdim_));
            negsrc = dot(hidden_data, weight.T());
            negsrc+=repmat(bias, neg_batchsize_);
            negsrc = F<op::sigmoid>(negsrc);
            for (int i = 0; i < neg_batchsize_; i++)
              for (int j = 0; j < vdim_; j++)
                negsrc[i][j] = static_cast<float>((rand() / static_cast<double>(RAND_MAX)) > negsrc[i][j] ? 0 : 1);
            hidden_data = dot(negsrc, weight);
        }
  }
}

void DBMBottomLayer::ComputeGradient(const vector<SLayer>& srclayers) {
  Tensor<cpu, 2> data(data_.mutable_cpu_data(), Shape2(batchsize_, hdim_));
  Tensor<cpu, 2> hidden_data(hidden_data_.mutable_cpu_data(),
              Shape2(neg_batchsize_, hdim_));
  kPhase = true;
  CHECK_EQ(srclayers[0]->data(this).count(), batchsize_*vdim_);
  Tensor<cpu, 2> possrc(srclayers[0]->mutable_data(this)->mutable_cpu_data(),
        Shape2(batchsize_, vdim_));
  Tensor<cpu, 2> negsrc(negsrc_.mutable_cpu_data(),
        Shape2(neg_batchsize_, vdim_));
  Tensor<cpu, 2> gweight(weight_->mutable_cpu_grad(), Shape2(vdim_, hdim_));
  Tensor<cpu, 1> gbias(bias_->mutable_cpu_grad(), Shape1(vdim_));
  gbias = sum_rows(negsrc);
  gbias-=sum_rows(possrc);
  gweight = dot(negsrc.T(), hidden_data);
  gweight-=dot(possrc.T(), data);
  gbias*=scale_/(1.0f*batchsize_);
  gweight*=scale_/(1.0f*batchsize_);
}

void DBMBottomLayer::ComputeLoss(Metric* perf) {
  float loss = (0.0f);
  kPhase = true;
  CHECK_EQ(srclayers_[0]->data(this).count(), batchsize_*vdim_);
  Tensor<cpu, 2> possrc(srclayers_[0]->mutable_data(this)->mutable_cpu_data(),
         Shape2(batchsize_, vdim_));
  Tensor<cpu, 2> data(data_.mutable_cpu_data(), Shape2(batchsize_, hdim_));
  Tensor<cpu, 2> weight(weight_->mutable_cpu_data(), Shape2(vdim_, hdim_));
  Tensor<cpu, 1> bias(bias_->mutable_cpu_data(), Shape1(vdim_));
  Tensor<cpu, 2> reconstruct(Shape2(batchsize_, vdim_)); /*reconstruct error*/
  AllocSpace(reconstruct);
  reconstruct = dot(data, weight.T());
  reconstruct+=repmat(bias, batchsize_);
  reconstruct = F<op::sigmoid>(reconstruct);
  for (int i = 0; i < batchsize_; i++)
    for (int j = 0; j < vdim_; j++) {
      loss += -(possrc[i][j]*log(reconstruct[i][j])+(1-possrc[i][j])*log(1-reconstruct[i][j]));
    }
  loss/=batchsize_;
  FreeSpace(reconstruct);
  perf->Reset();
  perf->AddMetric("reconstruct_error", loss);
}
/**************** Implementation for DBMTopLayer********************/
void DBMTopLayer::Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers) {
  CHECK_EQ(srclayers.size(), 1);
  kPhase = true;
  const auto& possrc = srclayers[0]->data(this);
  kPhase = false;
  const auto& negsrc = srclayers[0]->data(this);
  is_first_iteration_top = true;
  scale_ = static_cast<float> (1.0f);
  batchsize_ = possrc.shape()[0];
  neg_batchsize_ = negsrc.shape()[0];
  vdim_ = possrc.count()/batchsize_;
  Factory<Param>* factory = Singleton<Factory<Param>>::Instance();
  bias_ = shared_ptr<Param>(factory->Create("Param"));
  bias_->Setup(proto.param(0), vector<int>{vdim_}, 0);
  srand((unsigned)time(NULL));
}

void DBMTopLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers) {
}

void DBMTopLayer::ComputeFeature(Phase phase, const vector<SLayer>& srclayers) {
  if (phase == kPositive) {  /*postive phase*/
    kPhase = true;
    CHECK_EQ(srclayers[0]->data(this).count(), batchsize_*vdim_);
    Tensor<cpu, 2> possrc(srclayers[0]->mutable_data(this)->mutable_cpu_data(),
       Shape2(batchsize_, vdim_));
    Tensor<cpu, 1> bias(bias_->mutable_cpu_data(), Shape1(vdim_));
    possrc+=repmat(bias, batchsize_);
    possrc = F<op::sigmoid>(possrc);
  }
  else if (phase == kNegative) {   /*negative phase*/
    kPhase = false;
    CHECK_EQ(srclayers[0]->data(this).count(), neg_batchsize_*vdim_);
    Tensor<cpu, 2> negsrc(srclayers[0]->mutable_data(this)->mutable_cpu_data(),
         Shape2(neg_batchsize_, vdim_));
    Tensor<cpu, 1> bias(bias_->mutable_cpu_data(), Shape1(vdim_));
    negsrc+=repmat(bias, neg_batchsize_);
    negsrc = F<op::sigmoid>(negsrc);
    for (int i = 0; i < neg_batchsize_; i++)
      for (int j = 0; j < vdim_; j++)
        negsrc[i][j] = static_cast<float>((rand() / static_cast<double>(RAND_MAX)) > negsrc[i][j] ? 0 : 1);
  }
  else if (phase == kTest) {   /*test phase*/
     kPhase = true;
     CHECK_EQ(srclayers[0]->data(this).count(), batchsize_*vdim_);
     Tensor<cpu, 2> possrc(srclayers[0]->mutable_data(this)->mutable_cpu_data(),
         Shape2(batchsize_, vdim_));
     for (int i = 0; i < batchsize_; i++)  /*the reconstruct vector*/
       for (int j = 0; j < vdim_; j++)
         possrc[i][j] = static_cast<float>((rand() / static_cast<double>(RAND_MAX)) > possrc[i][j] ? 0 : 1);
  }
}

void DBMTopLayer::ComputeGradient(const vector<SLayer>& srclayers) {
  kPhase = true;
  CHECK_EQ(srclayers[0]->data(this).count(), batchsize_*vdim_);
  Tensor<cpu, 2> possrc(srclayers[0]->mutable_data(this)->mutable_cpu_data(),
    Shape2(batchsize_, vdim_));
  kPhase = false;
  CHECK_EQ(srclayers[0]->data(this).count(), neg_batchsize_*vdim_);
  Tensor<cpu, 2> negsrc(srclayers[0]->mutable_data(this)->mutable_cpu_data(),
       Shape2(neg_batchsize_, vdim_));
  Tensor<cpu, 1> gbias(bias_->mutable_cpu_grad(), Shape1(vdim_));
  gbias = sum_rows(negsrc);
  gbias-=sum_rows(possrc);
  gbias*=scale_/(1.0f*batchsize_);
}
/**************** Implementation for InnerProductLayer********************/
void InnerProductLayer::Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  const auto& src=srclayers[0]->data(this);
  batchsize_=src.shape()[0];
  vdim_=src.count()/batchsize_;
  hdim_=proto.innerproduct_conf().num_output();
  data_.Reshape(vector<int>{batchsize_, hdim_});
  grad_.ReshapeLike(data_);
  Factory<Param>* factory=Singleton<Factory<Param>>::Instance();
  weight_=shared_ptr<Param>(factory->Create("Param"));
  bias_=shared_ptr<Param>(factory->Create("Param"));
  weight_->Setup(proto.param(0), vector<int>{vdim_, hdim_});
  bias_->Setup(proto.param(1), vector<int>{hdim_});
}
void InnerProductLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  LayerProto newproto(proto);
  InnerProductProto * innerproto=newproto.mutable_innerproduct_conf();
  innerproto->set_num_output(shape[1]);
  Setup(newproto, srclayers);
}

void InnerProductLayer::ComputeFeature(Phase phase, const vector<SLayer>& srclayers) {
  Tensor<cpu, 2> data(data_.mutable_cpu_data(), Shape2(batchsize_,hdim_));
  CHECK_EQ(srclayers[0]->data(this).count(), batchsize_*vdim_);
  Tensor<cpu, 2> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(),
      Shape2(batchsize_,vdim_));
  Tensor<cpu, 2> weight(weight_->mutable_cpu_data(), Shape2(vdim_,hdim_));
  Tensor<cpu, 1> bias(bias_->mutable_cpu_data(), Shape1(hdim_));
  data=dot(src, weight);
  // repmat: repeat bias vector into batchsize rows
  data+=repmat(bias, batchsize_);
}

void InnerProductLayer::ComputeGradient(const vector<SLayer>& srclayers) {
  Tensor<cpu, 2> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(),
      Shape2(batchsize_,vdim_));
  Tensor<cpu, 2> grad(grad_.mutable_cpu_data(),Shape2(batchsize_,hdim_));
  Tensor<cpu, 2> weight(weight_->mutable_cpu_data(), Shape2(vdim_,hdim_));
  Tensor<cpu, 2> gweight(weight_->mutable_cpu_grad(), Shape2(vdim_,hdim_));
  Tensor<cpu, 1> gbias(bias_->mutable_cpu_grad(), Shape1(hdim_));

  gbias=sum_rows(grad);
  gweight=dot(src.T(), grad);
  if(srclayers[0]->mutable_grad(this)!=nullptr){
    Tensor<cpu, 2> gsrc(srclayers[0]->mutable_grad(this)->mutable_cpu_data(),
        Shape2(batchsize_,vdim_));
    gsrc=dot(grad, weight.T());
  }
}
/*****************************************************************************
 * Implementation for LabelLayer
 *****************************************************************************/
void LabelLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  int batchsize=static_cast<DataLayer*>(srclayers[0].get())->batchsize();
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


/*********************LMDBDataLayer**********************************/
void LMDBDataLayer::ComputeFeature(Phase phase, const vector<SLayer>& srclayers){
  if(random_skip_){
    int nskip=rand()%random_skip_;
    int n=0;
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
          &mdb_value_, MDB_FIRST), MDB_SUCCESS);
    while (mdb_cursor_get(mdb_cursor_, &mdb_key_,
          &mdb_value_, MDB_NEXT) == MDB_SUCCESS)
      n++;
    LOG(INFO)<<"Random Skip "<<nskip<<" records of total "<<n<<"records";
    // We have reached the end. Restart from the first.
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
          &mdb_value_, MDB_FIRST), MDB_SUCCESS);
    for(int i=0;i<nskip;i++){
      if (mdb_cursor_get(mdb_cursor_, &mdb_key_,
            &mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_FIRST), MDB_SUCCESS);
      }
    }
    random_skip_=0;
  }
  Datum datum;
  for(auto& record: records_){
    SingleLabelImageRecord* image=record.mutable_image();
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
          &mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
    datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    ConvertDatumToSingleLableImageRecord(datum, image);
    if (mdb_cursor_get(mdb_cursor_, &mdb_key_,
          &mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
            &mdb_value_, MDB_FIRST), MDB_SUCCESS);
    }
  }
}

void LMDBDataLayer::ConvertDatumToSingleLableImageRecord(const Datum& datum,
    SingleLabelImageRecord* record){
  record->set_label(datum.label());
  record->clear_shape();
  if(datum.has_channels())
    record->add_shape(datum.channels());
  if(datum.has_height())
    record->add_shape(datum.height());
  if(datum.has_width())
    record->add_shape(datum.width());
  if(datum.has_data())
    record->set_pixel(datum.data());
  if(datum.float_data_size()){
    record->clear_data();
    for(float x: datum.float_data())
      record->add_data(x);
  }
}

void LMDBDataLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
  CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS); // 1TB
  CHECK_EQ(mdb_env_open(mdb_env_,
        proto.lmdbdata_conf().path().c_str(),
        MDB_RDONLY, 0664), MDB_SUCCESS) << "cannot open lmdb "
    << proto.lmdbdata_conf().path();
  CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
    << "mdb_txn_begin failed";
  CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
    << "mdb_open failed";
  CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
    << "mdb_cursor_open failed";
  LOG(INFO) << "Opening lmdb " << proto.lmdbdata_conf().path();
  CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
      MDB_SUCCESS) << "mdb_cursor_get failed";

  if (mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT)
      != MDB_SUCCESS) {
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_,
          MDB_FIRST), MDB_SUCCESS);
  }
  Datum datum;
  datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
  SingleLabelImageRecord* record=sample_.mutable_image();
  ConvertDatumToSingleLableImageRecord(datum, record);

  batchsize_=batchsize();
  records_.resize(batchsize_);
  random_skip_=proto.lmdbdata_conf().random_skip();
}

/***************** Implementation for LRNLayer *************************/
void LRNLayer::Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  lsize_ = proto.lrn_conf().local_size();
  CHECK_EQ(lsize_ % 2, 1) << "LRN only supports odd values for Localvol";
  knorm_=proto.lrn_conf().knorm();
  alpha_ = proto.lrn_conf().alpha();
  beta_ = proto.lrn_conf().beta();

  const vector<int>& s=srclayers[0]->data(this).shape();
  data_.Reshape(s);
  grad_.Reshape(s);
  norm_.Reshape(s);
  batchsize_=s[0];
  channels_=s[1];
  height_=s[2];
  width_=s[3];
}

void LRNLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  Setup(proto, srclayers);
}

void LRNLayer::ComputeFeature(Phase phase, const vector<SLayer>& srclayers){
  const float salpha = alpha_ / lsize_;
  Shape<4> s=Shape4(batchsize_,channels_, height_, width_);
  Tensor<cpu, 4> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(), s);
  Tensor<cpu, 4> data(data_.mutable_cpu_data(), s);
  Tensor<cpu, 4> norm(norm_.mutable_cpu_data(), s);
  // stores normalizer without power
  norm= chpool<red::sum>( F<op::square>(src) , lsize_ ) * salpha + knorm_;
  data = src * F<op::power>(norm, -beta_ );
}

void LRNLayer::ComputeGradient(const vector<SLayer>& srclayers) {
  const float salpha = alpha_ / lsize_;
  Shape<4> s=Shape4(batchsize_,channels_, height_, width_);
  Tensor<cpu, 4> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(), s);
  Tensor<cpu, 4> norm(norm_.mutable_cpu_data(), s);
  Tensor<cpu, 4> grad(grad_.mutable_cpu_data(), s);
  Tensor<cpu, 4> gsrc(srclayers[0]->mutable_grad(this)->mutable_cpu_data(), s);

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

  float* dptr=blob->mutable_cpu_data();
  for(const Record& record: records){
    // copy from record to cv::Mat
    cv::Mat input(inputsize, inputsize, CV_32FC1);
    const SingleLabelImageRecord& imagerecord=record.image();
    if(imagerecord.pixel().size()){
      string pixel=imagerecord.pixel();
      for(int i=0,k=0;i<inputsize;i++)
        for(int j=0;j<inputsize;j++)
          // NOTE!!! must cast pixel to uint8_t then to float!!! waste a lot of
          // time to debug this
          input.at<float>(i,j)=static_cast<float>(static_cast<uint8_t>(pixel[k++]));
    }else{
      for(int i=0,k=0;i<inputsize;i++)
        for(int j=0;j<inputsize;j++)
          input.at<float>(i,j)=imagerecord.data(k++);
    }
    int size=blob->shape()[1];
    /*
    cv::Mat resizeMat=input;
    // affine transform, scaling, rotation and shearing
    if(gamma_){
      float r1=rand_real()*2-1;
      float r2=rand_real()*2-1;
      int h=static_cast<int>(inputsize*(1.+r1*gamma_/100.0));
      int w=static_cast<int>(inputsize*(1.+r2*gamma_/100.0));
      cv::resize(input, resizeMat, cv::Size(h,w));
    }
    cv::Mat betaMat=resizeMat;
    cv::Mat warpmat(2,3, CV_32FC1);
    warpmat.at<float>(0,0)=1.0;
    warpmat.at<float>(0,1)=0.0;
    warpmat.at<float>(0,2)=0.0;
    warpmat.at<float>(1,0)=0.0;
    warpmat.at<float>(1,1)=1.0;
    warpmat.at<float>(1,2)=0.0;

    if(beta_){
      float r=rand_real()*2-1;
      if(rand() % 2){ // rotation
        cv::Point center(resizeMat.rows/2, resizeMat.cols/2);
        warpmat=cv::getRotationMatrix2D(center, r*beta_, 1.0);
      }else{
        //shearing
        warpmat.at<float>(0,1)=r*beta_/90;
        if(imagerecord.label()==1 ||imagerecord.label()==7)
          warpmat.at<float>(0,1)/=2.0;
      }
    }
    cv::warpAffine(resizeMat, betaMat, warpmat, cv::Size(size, size));
    */

    for(int i=0;i<size;i++){
      for(int j=0;j<size;j++){
        *dptr=input.at<float>(i,j)/norm_a_-norm_b_;
        dptr++;
      }
    }
  }
  CHECK_EQ(dptr, blob->mutable_cpu_data()+blob->count());
}
void MnistLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  int batchsize=static_cast<DataLayer*>(srclayers[0].get())->batchsize();
  Record sample=static_cast<DataLayer*>(srclayers[0].get())->sample();
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
void PoolingLayer::Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  PoolingProto pool_conf = proto.pooling_conf();
  kernel_=pool_conf.kernel();
  stride_=pool_conf.stride();
  CHECK_LT(pad_, kernel_);
  pool_=proto.pooling_conf().pool();
  CHECK(pool_ == PoolingProto_PoolMethod_AVE
        || pool_ == PoolingProto_PoolMethod_MAX)
      << "Padding implemented only for average and max pooling.";

  const auto& srcshape=srclayers[0]->data(this).shape();
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

void PoolingLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  Setup(proto, srclayers);
}

void PoolingLayer::ComputeFeature(Phase phase, const vector<SLayer>& srclayers){
  Tensor<cpu, 4> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(),
      Shape4(batchsize_, channels_, height_, width_));
  Tensor<cpu, 4> data(data_.mutable_cpu_data(),
      Shape4(batchsize_, channels_, pooled_height_, pooled_width_));
  if(pool_ == PoolingProto_PoolMethod_MAX)
    data=pool<red::maximum>(src, kernel_, stride_);
  else if(pool_ == PoolingProto_PoolMethod_AVE)
    data=pool<red::sum>(src, kernel_, stride_)
      *(1.0f/(kernel_*kernel_));
}

/*
 * partition only on num/channel dim
 * assume grad and data have the same paritition
 */
void PoolingLayer::ComputeGradient(const vector<SLayer>& srclayers) {
  Shape<4> s1= Shape4(batchsize_, channels_, height_, width_);
  Tensor<cpu, 4> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(),s1);
  Tensor<cpu, 4> gsrc(srclayers[0]->mutable_grad(this)->mutable_cpu_data(),s1);
  Shape<4> s2= Shape4(batchsize_, channels_, pooled_height_, pooled_width_);
  Tensor<cpu, 4> data(data_.mutable_cpu_data(), s2);
  Tensor<cpu, 4> grad(grad_.mutable_cpu_data(), s2);
  if(pool_ == PoolingProto_PoolMethod_MAX)
      gsrc = unpool<red::maximum>(src, data, grad, kernel_, stride_);
  else if(pool_ == PoolingProto_PoolMethod_AVE)
      gsrc = unpool<red::sum>(src, data, grad, kernel_, stride_)
        *(1.0f/(kernel_*kernel_));
}

/***************** Implementation for ReLULayer *****************************/

void ReLULayer::Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers){
  data_.ReshapeLike(srclayers[0]->data(this));
  grad_.ReshapeLike(*(srclayers[0]->mutable_grad(this)));
}

void ReLULayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  Setup(proto, srclayers);
}

void ReLULayer::ComputeFeature(Phase phase, const vector<SLayer>& srclayers){
  Tensor<cpu, 1> data(data_.mutable_cpu_data(), Shape1(data_.count()));
  Tensor<cpu, 1> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(),
      Shape1(data_.count()));
  data=F<op::relu>(src);
}

void ReLULayer::ComputeGradient(const vector<SLayer>& srclayers) {
  Tensor<cpu, 1> grad(grad_.mutable_cpu_data(), Shape1(grad_.count()));
  Tensor<cpu, 1> data(data_.mutable_cpu_data(), Shape1(data_.count()));
  Tensor<cpu, 1> gsrc(srclayers[0]->mutable_grad(this)->mutable_cpu_data(),
      Shape1(data_.count()));
  gsrc=F<op::relu_grad>(data)*grad;
}

/*************** Implementation for RGBImageLayer *************************/

void RGBImageLayer::ParseRecords(Phase phase,
    const vector<Record>& records, Blob<float>* blob){
  const vector<int>& s=blob->shape();
  Tensor<cpu, 4> images(data_.mutable_cpu_data(), Shape4(s[0],s[1],s[2],s[3]));
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
void RGBImageLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  scale_=proto.rgbimage_conf().scale();
  cropsize_=proto.rgbimage_conf().cropsize();
  mirror_=proto.rgbimage_conf().mirror();
  int batchsize=static_cast<DataLayer*>(srclayers[0].get())->batchsize();
  Record sample=static_cast<DataLayer*>(srclayers[0].get())->sample();
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
    if(proto.rgbimage_conf().meanfile().find("binaryproto")!=string::npos){
      BlobProto tmp;
      ReadProtoFromBinaryFile(proto.rgbimage_conf().meanfile().c_str(), &tmp);
      CHECK_EQ(mean_.count(), tmp.data_size());
      memcpy(mean_.mutable_cpu_data(), tmp.data().data(), sizeof(float)*tmp.data_size());
    }else{
      SingleLabelImageRecord tmp;
      ReadProtoFromBinaryFile(proto.rgbimage_conf().meanfile().c_str(), &tmp);
      CHECK_EQ(mean_.count(), tmp.data_size());
      memcpy(mean_.mutable_cpu_data(), tmp.data().data(), sizeof(float)*tmp.data_size());
    }
  }else{
    memset(mean_.mutable_cpu_data(),0,sizeof(float)*mean_.count());
  }
}

/***************Implementation for ShardDataLayer**************************/
void ShardDataLayer::ComputeFeature(Phase phase, const vector<SLayer>& srclayers){
  if(random_skip_){
    int nskip=rand()%random_skip_;
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

void ShardDataLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  shard_= std::make_shared<DataShard>(proto.sharddata_conf().path(),
      DataShard::kRead);
  string key;
  shard_->Next(&key, &sample_);
  batchsize_=proto.sharddata_conf().batchsize();

  records_.resize(batchsize_);
  random_skip_=proto.sharddata_conf().random_skip();
}
/*******************Implementation of TanLayer***************************/
void TanhLayer::Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers){
  data_.ReshapeLike(srclayers[0]->data(this));
  grad_.ReshapeLike(srclayers[0]->grad(this));
}

void TanhLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  Setup(proto, srclayers);
}


void TanhLayer::ComputeFeature(Phase phase, const vector<SLayer>& srclayers){
  Tensor<cpu, 1> data(data_.mutable_cpu_data(), Shape1(data_.count()));
  Tensor<cpu, 1> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(),
      Shape1(data_.count()));
  data=F<op::stanh>(src);
}

void TanhLayer::ComputeGradient(const vector<SLayer>& srclayers) {
  Tensor<cpu, 1> data(data_.mutable_cpu_data(), Shape1(data_.count()));
  Tensor<cpu, 1> grad(grad_.mutable_cpu_data(), Shape1(grad_.count()));
  Tensor<cpu, 1> gsrc(srclayers[0]->mutable_grad(this)->mutable_cpu_data(),
      Shape1(data_.count()));
  gsrc=F<op::stanh_grad>(data)*grad;
}
/********** * Implementation for SoftmaxLossLayer*************************/
void SoftmaxLossLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),2);
  data_.Reshape(srclayers[0]->data(this).shape());
  batchsize_=data_.shape()[0];
  dim_=data_.count()/batchsize_;
  topk_=proto.softmaxloss_conf().topk();
  metric_.Reshape(vector<int>{2});
  scale_=proto.softmaxloss_conf().scale();
}
void SoftmaxLossLayer::SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){
  Setup(proto, srclayers);
}
void SoftmaxLossLayer::ComputeFeature(Phase phase, const vector<SLayer>& srclayers) {
  Shape<2> s=Shape2(batchsize_, dim_);
  Tensor<cpu, 2> prob(data_.mutable_cpu_data(), s);
  Tensor<cpu, 2> src(srclayers[0]->mutable_data(this)->mutable_cpu_data(), s);
  Softmax(prob, src);
  const float* label=srclayers[1]->data(this).cpu_data();
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
  float *metric=metric_.mutable_cpu_data();
  metric[0]=loss*scale_/(1.0f*batchsize_);
  metric[1]=precision*scale_/(1.0f*batchsize_);
}

void SoftmaxLossLayer::ComputeGradient(const vector<SLayer>& srclayers) {
  const float* label=srclayers[1]->data(this).cpu_data();
  Blob<float>* gsrcblob=srclayers[0]->mutable_grad(this);
  gsrcblob->CopyFrom(data_);
  float* gsrcptr=gsrcblob->mutable_cpu_data();
  for(int n=0;n<batchsize_;n++){
    gsrcptr[n*dim_+static_cast<int>(label[n])]-=1.0f;
  }
  Tensor<cpu, 1> gsrc(gsrcptr, Shape1(gsrcblob->count()));
  gsrc*=scale_/(1.0f*batchsize_);
}

}  // namespace singa
