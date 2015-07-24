#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cblas.h>
#include <math.h>
#include <cfloat>
#include <glog/logging.h>
#include "utils/singleton.h"
#include "utils/factory.h"
#include "neuralnet/base_layer.h"

namespace singa {

void Layer::Setup(const LayerProto& proto, int npartitions) {
  CHECK_GE(npartitions, 1);
  layer_proto_ = proto;
}

const string Layer::DebugString(int step, Phase phase) {
  string ret =StringPrintf("Layer %10s ", name().c_str());
  if(data_.count() != 0)
    return ret;
  if(phase == kForward) {
    ret += StringPrintf("data %10s data norm1 %13.9f", data_.asum_data());
  }else if(phase == kBackward) {
    ret += StringPrintf("grad norm1 %13.9f\n", grad_.asum_data());
    for(Param* p: GetParams())
      ret += StringPrintf("param id %2d, name %10s,\
          value norm1 %13.9f, grad norm1 %13.9f\n",
          p->id(), p->name().c_str(),
          p->data().asum_data(), p->grad().asum_data());
  }
  return ret;
}
/********* Implementation for BridgeDstLayer **************/
void BridgeDstLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(srclayers_.size(),1);
  data_.Reshape(srclayers_[0]->data(this).shape());
  grad_.ReshapeLike(data_);
}

/************* Implementation for ConcateLayer ***********/
void ConcateLayer::Setup(const LayerProto& proto, int npartitions) {
  // CHECK_EQ(npartitions, 1);
  Layer::Setup(proto, npartitions);
  size_t concate_dim=proto.concate_conf().concate_dim();
  CHECK_GE(concate_dim,0);
  CHECK_GT(srclayers_.size(),1);
  vector<int> shape=srclayers_[0]->data(this).shape();
  for(size_t i=1;i<srclayers_.size();i++){
    const vector<int>& srcshape=srclayers_[i]->data(this).shape();
    for(size_t j=0;j<shape.size();j++)
      if(j==concate_dim)
        shape[j]+=srcshape[j];
      else
        CHECK_EQ(shape[j], srcshape[j]);
  }
  data_.Reshape(shape);
  grad_.Reshape(shape);
}

void ConcateLayer::ComputeFeature(Phase phase, Metric *perf){
  LOG(FATAL) << "Not implemented for Concate Layer";
}

void ConcateLayer::ComputeGradient(Phase phase){
  LOG(FATAL) << "Not implemented for Concate Layer";
}

/************* Implementation for ParserLayer ***********/
void ParserLayer::ComputeFeature(Phase phase, Metric *perf){
  CHECK_EQ(srclayers_.size(),1);
  auto datalayer=static_cast<DataLayer*>(*srclayers_.begin());
  ParseRecords(phase, datalayer->records(), &data_);
}

/************* Implementation for PrefetchLayer ***********/
void PrefetchLayer::Prefetch(Phase phase){
  //clock_t s=clock();
  for(auto layer: sublayers_)
    layer->ComputeFeature(phase, nullptr);
  //LOG(ERROR)<<(clock()-s)*1.0/CLOCKS_PER_SEC;
}

void PrefetchLayer::ComputeFeature(Phase phase, Metric* perf){
  if(thread_.joinable())
    thread_.join();
  else{
    Prefetch(phase);
  }
  for(auto layer: sublayers_){
    if(layer->is_parserlayer())
      // TODO replace CopyFrom with Swap?
      datablobs_.at(layer->name()).CopyFrom(layer->data(this));
  }
  thread_=std::thread(&PrefetchLayer::Prefetch, this, phase);
}

void PrefetchLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  // CHECK_EQ(npartitions, 1);
  Factory<Layer>* factory=Singleton<Factory<Layer>>::Instance();
  const auto& sublayers=proto.prefetch_conf().sublayers();
  CHECK_GE(sublayers.size(), 1);
  map<string, Layer*> layers;
  for(auto const &p:sublayers){
    auto layer=factory->Create(p.type());
    sublayers_.push_back(layer);
    layers[p.name()]= layer;
  }
  // TODO topology sort layers
  auto layer=sublayers_.begin();
  for(auto const &p : sublayers){
    std::vector<Layer*> src;
    for(auto const &srcname: p.srclayers()){
      src.push_back(layers[srcname]);
      (*layer)->add_srclayer(layers[srcname]);
    }
    (*layer)->Setup(p);
    layer++;
  }
  for(auto layer: sublayers_)
    if(layer->is_parserlayer())
      datablobs_[layer->name()]=Blob<float>(layer->data(this).shape());
}

const Blob<float>& PrefetchLayer::data(const Layer* from, Phase phase) const {
  if(from!=nullptr){
    return datablobs_.at(from->datablob());
  }else{
    //CHECK_EQ(datablobs_.size(),1);
    return datablobs_.begin()->second;
  }
}

Blob<float>* PrefetchLayer::mutable_data(const Layer* from, Phase phase) {
  if(from!=nullptr){
    return &(datablobs_.at(from->datablob()));
  }else{
    //CHECK_EQ(datablobs_.size(),1);
    return &(datablobs_.begin()->second);
  }
}

PrefetchLayer::~PrefetchLayer(){
  if(thread_.joinable())
    thread_.join();
  for(auto layer : sublayers_)
    delete layer;
}
/************* Implementation for SliceLayer****************/
void SliceLayer::Setup(const LayerProto& proto, int npartitions){
  // CHECK_EQ(npartitions, 1);
  Layer::Setup(proto, npartitions);
  slice_dim_=proto.slice_conf().slice_dim();
  slice_num_= npartitions;
  CHECK_GE(slice_dim_,0);
  CHECK_EQ(slice_num_, dstlayers_.size());
  data_.Reshape(srclayers_[0]->data(this).shape());
  grad_.ReshapeLike(data_);
  datavec_.resize(slice_num_);
  gradvec_.resize(slice_num_);
  CHECK_EQ(data_.count()%slice_num_, 0); // restrict equal slicing
  //LOG(ERROR)<<"slice dim "<<slice_dim<<" slice num "<<slice_num;
  for(int i=0;i<slice_num_;i++){
    vector<int> newshape(data_.shape());
    newshape[slice_dim_]=newshape[slice_dim_]/slice_num_+
      ((i==slice_num_-1)?newshape[slice_dim_]%slice_num_:0);
    datavec_[i].Reshape(newshape);
    gradvec_[i].Reshape(newshape);
    //LOG(ERROR)<<"slice "<<IntVecToString(newshape);
  }
}

int SliceLayer::SliceID(const Layer* layer) const {
  CHECK(layer!= nullptr);
  for(size_t i=0;i<datavec_.size();i++){
    //LOG(ERROR)<<"get slice "<<IntVecToString(shapes_[i]);
    if(dstlayers_[i] == layer)
      return i;
  }
  CHECK(false);
  return -1;
}

const Blob<float>& SliceLayer::data(const Layer* layer, Phase phase) const {
  if(layer==nullptr)
    return data_;
  return datavec_[SliceID(layer)];
}
const Blob<float>& SliceLayer::grad(const Layer* layer) const {
  if(layer==nullptr)
    return grad_;
  return gradvec_[SliceID(layer)];
}
Blob<float>* SliceLayer::mutable_data(const Layer* layer, Phase phase) {
  if(layer==nullptr)
    return &data_;
  return &datavec_[SliceID(layer)];
}
Blob<float>* SliceLayer::mutable_grad(const Layer* layer){
  if(layer==nullptr)
    return &grad_;
  return &gradvec_[SliceID(layer)];
}
void SliceLayer::ComputeFeature(Phase phase, Metric *perf) {
  CHECK_EQ(srclayers_.size(),1);
  if(slice_dim_==0){
    const auto& blob=srclayers_.at(0)->data(this);
    int size=blob.count()/slice_num_;
    for(int i=0;i<slice_num_;i++){
      float* dst=datavec_[i].mutable_cpu_data();
      const float* src=blob.cpu_data()+i*size;
      memcpy(dst, src, size*sizeof(float));
    }
  }
}
void SliceLayer::ComputeGradient(Phase phase) {
  // LOG(FATAL) << "Not implemented";
}

/************* Implementation for SplitLayer****************/
void SplitLayer::Setup(const LayerProto& proto, int npartitions) {
  // CHECK_EQ(npartitions, 1);
  Layer::Setup(proto, npartitions);

  CHECK_EQ(srclayers_.size(),1);
  data_.Reshape(srclayers_[0]->data(this).shape());
  grad_.Reshape(srclayers_[0]->data(this).shape());
}

void SplitLayer::ComputeFeature(Phase phase, Metric *perf) {
  LOG(FATAL) << "Not implemented";

}
void SplitLayer::ComputeGradient(Phase phase) {
  LOG(FATAL) << "Not implemented";
}

}  // namespace singa

