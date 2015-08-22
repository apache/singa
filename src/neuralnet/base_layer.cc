#include "neuralnet/base_layer.h"

#include <cblas.h>
#include <glog/logging.h>
#include <math.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cfloat>
#include "utils/factory.h"
#include "utils/singleton.h"

namespace singa {

using std::string;
using std::vector;

Layer* Layer::Create(const LayerProto& proto) {
  auto* factory = Singleton<Factory<Layer>>::Instance();
  Layer* layer = nullptr;
  if (proto.has_user_type())
    layer = factory->Create(proto.user_type());
  else
    layer = factory->Create(proto.type());
  return layer;
}

const string Layer::DebugString(int step, Phase phase) {
  string ret = StringPrintf("Layer %10s ", name().c_str());
  if (data_.count() != 0)
    return ret;
  if (phase == kForward) {
    ret += StringPrintf("data %10s data norm1 %13.9f", data_.asum_data());
  } else if (phase == kBackward) {
    ret += StringPrintf("grad norm1 %13.9f\n", grad_.asum_data());
    for (Param* p : GetParams()) {
      ret += StringPrintf(
          "param id %2d, name %10s, value norm1 %13.9f, grad norm1 %13.9f\n",
          p->id(), p->name().c_str(), p->data().asum_data(),
          p->grad().asum_data());
    }
  }
  return ret;
}

/************* Implementation for ParserLayer ***********/
void ParserLayer::ComputeFeature(Phase phase, Metric *perf) {
  CHECK_EQ(srclayers_.size(), 1);
  auto datalayer = static_cast<DataLayer*>(*srclayers_.begin());
  ParseRecords(phase, datalayer->records(), &data_);
}

/************* Implementation for PrefetchLayer ***********/
PrefetchLayer::~PrefetchLayer() {
  if (thread_.joinable())
    thread_.join();
  for (auto layer : sublayers_)
    delete layer;
}

void PrefetchLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  // CHECK_EQ(npartitions, 1);
  Factory<Layer>* factory = Singleton<Factory<Layer>>::Instance();
  const auto& sublayers = proto.prefetch_conf().sublayers();
  CHECK_GE(sublayers.size(), 1);
  std::map<string, Layer*> layers;
  for (auto const &p : sublayers) {
    auto layer = factory->Create(p.type());
    sublayers_.push_back(layer);
    layers[p.name()] = layer;
  }
  // TODO(wangwei) topology sort layers
  auto layer = sublayers_.begin();
  for (auto const &p : sublayers) {
    std::vector<Layer*> src;
    for (auto const &srcname : p.srclayers()) {
      src.push_back(layers[srcname]);
      (*layer)->add_srclayer(layers[srcname]);
    }
    (*layer)->Setup(p);
    layer++;
  }
  for (auto layer : sublayers_)
    if (layer->is_parserlayer())
      datablobs_[layer->name()] = Blob<float>(layer->data(this).shape());
}

void PrefetchLayer::ComputeFeature(Phase phase, Metric* perf) {
  if (thread_.joinable())
    thread_.join();
  else
    Prefetch(phase);
  for (auto layer : sublayers_) {
    if (layer->is_parserlayer())
      // TODO(wangwei) replace CopyFrom with Swap?
      datablobs_.at(layer->name()).CopyFrom(layer->data(this));
  }
  thread_ = std::thread(&PrefetchLayer::Prefetch, this, phase);
}

void PrefetchLayer::Prefetch(Phase phase) {
  // clock_t s=clock();
  for (auto layer : sublayers_)
    layer->ComputeFeature(phase, nullptr);
  // LOG(ERROR)<<(clock()-s)*1.0/CLOCKS_PER_SEC;
}

const Blob<float>& PrefetchLayer::data(const Layer* from) const {
  LOG(FATAL) << " needs update";
  if (from != nullptr) {
    return datablobs_.at("");
  } else {
    // CHECK_EQ(datablobs_.size(),1);
    return datablobs_.begin()->second;
  }
}

Blob<float>* PrefetchLayer::mutable_data(const Layer* from) {
  LOG(FATAL) << " needs update";
  if (from != nullptr) {
    return &(datablobs_.at(""));
  } else {
    // CHECK_EQ(datablobs_.size(),1);
    return &(datablobs_.begin()->second);
  }
}

}  // namespace singa
