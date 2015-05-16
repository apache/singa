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
/********* Implementation for Layer **************/
void Layer::Init(const LayerProto &proto) {
  layer_proto_=proto;
}

void Layer::Init(const Layer& other, const vector<int>& shape){
  data_.Reshape(shape);
  grad_.Reshape(shape);
  layer_proto_=other.layer_proto_;
}
void Layer::Setup(){
  Setup(layer_proto_, srclayers_);
}
void Layer::SetupAfterPartition(){
  vector<int> shape=data_.shape();
  SetupAfterPartition(layer_proto_, shape, srclayers_);
  //LOG(ERROR)<<name()<<":"<<IntVecToString(shape_);
  CHECK(std::equal(shape.begin(), shape.end(), data_.shape().begin()))<<name()
    <<IntVecToString(shape)<<"--"<<IntVecToString(data_.shape());
}
void Layer::ComputeFeature(bool training){
  ComputeFeature(training, srclayers_);
}
void Layer::ComputeGradient(){
  ComputeGradient(srclayers_);
}

void Layer::ToProto(LayerProto *proto, bool copyData) {
}

/********* Implementation for BridgeSrcLayer **************/
void BridgeSrcLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  data_.Reshape(srclayers[0]->data(this).shape());
  grad_.ReshapeLike(data_);
}
void BridgeSrcLayer::SetupAfterPartition(){
  Setup(layer_proto_, srclayers_);
  //LOG(ERROR)<<name()<<":"<<IntVecToString(shape_);
}

void BridgeSrcLayer::ComputeFeature(bool training,
    const vector<SLayer>& srclayers){
}
void BridgeSrcLayer::ComputeGradient(const vector<SLayer>& srclayers){
}

/********* Implementation for BridgeDstLayer **************/
void BridgeDstLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  data_.Reshape(srclayers[0]->data(this).shape());
  grad_.ReshapeLike(data_);
}
void BridgeDstLayer::SetupAfterPartition(){
  Setup(layer_proto_, srclayers_);
  //LOG(ERROR)<<name()<<":"<<IntVecToString(shape_);
}


/************* Implementation for ConcateLayer ***********/
void ConcateLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  size_t concate_dim=proto.concate_param().concate_dimension();
  CHECK_GE(concate_dim,0);
  CHECK_GT(srclayers.size(),1);
  vector<int> shape=srclayers[0]->data(this).shape();
  for(size_t i=1;i<srclayers.size();i++){
    const vector<int>& srcshape=srclayers[i]->data(this).shape();
    for(size_t j=0;j<shape.size();j++)
      if(j==concate_dim)
        shape[j]+=srcshape[j];
      else
        CHECK_EQ(shape[j], srcshape[j]);
  }
  data_.Reshape(shape);
  grad_.Reshape(shape);
}

void ConcateLayer::SetupAfterPartition(){
  Setup(layer_proto_, srclayers_);
//  LOG(ERROR)<<name()<<":"<<IntVecToString(shape_);
}

void ConcateLayer::ComputeFeature(bool training, const vector<SLayer>& srclayers){}

void ConcateLayer::ComputeGradient(const vector<shared_ptr<Layer>>& srclayers){}

/************* Implementation for ParserLayer ***********/
void ParserLayer::ComputeFeature(bool training, const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  auto datalayer=static_cast<DataLayer*>(srclayers.begin()->get());
  ParseRecords(training, datalayer->records(), &data_);
}

/************* Implementation for PrefetchLayer ***********/
void PrefetchLayer::Prefetch(bool training){
  //clock_t s=clock();
  for(auto layer: sublayers_)
    layer->ComputeFeature(training);
  //LOG(ERROR)<<(clock()-s)*1.0/CLOCKS_PER_SEC;
}

void PrefetchLayer::ComputeFeature(bool training,
    const vector<SLayer>& srclayers){
  if(thread_.joinable())
    thread_.join();
  else{
    Prefetch(training);
  }
  for(auto layer: sublayers_){
    if(layer->is_parserlayer())
      // TODO replace CopyFrom with Swap?
      datablobs_.at(layer->name()).CopyFrom(layer->data(this));
  }
  thread_=std::thread(&PrefetchLayer::Prefetch, this, training);
}

void PrefetchLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  Factory<Layer>* factory=Singleton<Factory<Layer>>::Instance();
  CHECK_GE(proto.sublayers_size(), 1);
  map<string, SLayer> layers;
  for(auto const &p:proto.sublayers()){
    auto layer=shared_ptr<Layer>(factory->Create(p.type()));
    layer->Init(p);
    sublayers_.push_back(layer);
    layers[p.name()]= layer;
  }
  // TODO topology sort layers
  auto layer=sublayers_.begin();
  for(auto const &p:proto.sublayers()){
    std::vector<SLayer> src;
    for(auto const &srcname: p.srclayers()){
      src.push_back(layers[srcname]);
      (*layer)->AddSrcLayer(layers[srcname]);
    }
    (*layer)->Setup(p, src);
    layer++;
  }
  for(auto layer: sublayers_)
    if(layer->is_parserlayer())
      datablobs_[layer->name()]=Blob<float>(layer->data(this).shape());
}

const Blob<float>& PrefetchLayer::data(const Layer* from) const {
  if(from!=nullptr){
    return datablobs_.at(from->datablob());
  }else{
    //CHECK_EQ(datablobs_.size(),1);
    return datablobs_.begin()->second;
  }
}

Blob<float>* PrefetchLayer::mutable_data(const Layer* from) {
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
}
/************* Implementation for SliceLayer****************/
void SliceLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  slice_dim_=proto.slice_param().slice_dimension();
  slice_num_=proto.slice_param().slice_num();
  CHECK_GE(slice_dim_,0);
  CHECK_EQ(slice_num_, dstlayers_.size());
  data_.Reshape(srclayers[0]->data(this).shape());
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

void SliceLayer::SetupAfterPartition(){
  Setup(layer_proto_, srclayers_);
  //LOG(ERROR)<<name()<<":"<<IntVecToString(shape_);
}


int SliceLayer::SliceID(const Layer* layer) const {
  CHECK(layer!= nullptr);
  for(size_t i=0;i<datavec_.size();i++){
    //LOG(ERROR)<<"get slice "<<IntVecToString(shapes_[i]);
    if(dstlayers_[i].get() == layer)
      return i;
  }
  CHECK(false);
  return -1;
}

const Blob<float>& SliceLayer::data(const Layer* layer) const {
  if(layer==nullptr)
    return data_;
  return datavec_[SliceID(layer)];
}
const Blob<float>& SliceLayer::grad(const Layer* layer) const {
  if(layer==nullptr)
    return grad_;
  return gradvec_[SliceID(layer)];
}
Blob<float>* SliceLayer::mutable_data(const Layer* layer) {
  if(layer==nullptr)
    return &data_;
  return &datavec_[SliceID(layer)];
}
Blob<float>* SliceLayer::mutable_grad(const Layer* layer){
  if(layer==nullptr)
    return &grad_;
  return &gradvec_[SliceID(layer)];
}
void SliceLayer::ComputeFeature(bool training,
    const vector<shared_ptr<Layer>>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  if(slice_dim_==0){
    const auto& blob=srclayers.at(0)->data(this);
    int size=blob.count()/slice_num_;
    for(int i=0;i<slice_num_;i++){
      float* dst=datavec_[i].mutable_cpu_data();
      const float* src=blob.cpu_data()+i*size;
      memcpy(dst, src, size*sizeof(float));
    }
  }
}
void SliceLayer::ComputeGradient(const vector<shared_ptr<Layer>>& srclayers){

}

void SplitLayer::Setup(const LayerProto& proto,
    const vector<SLayer>& srclayers){
  CHECK_EQ(srclayers.size(),1);
  data_.Reshape(srclayers[0]->data(this).shape());
  grad_.Reshape(srclayers[0]->data(this).shape());
}

/************* Implementation for SplitLayer****************/
void SplitLayer::SetupAfterPartition(){
  Setup(layer_proto_, srclayers_);
  //LOG(ERROR)<<name()<<":"<<IntVecToString(shape_);
}
void SplitLayer::ComputeFeature(bool training, const vector<shared_ptr<Layer>>& srclayers){

}
void SplitLayer::ComputeGradient(const vector<shared_ptr<Layer>>& srclayers){

}

}  // namespace singa

