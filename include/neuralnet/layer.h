#ifndef SINGA_NEURALNET_LAYER_H_
#define SINGA_NEURALNET_LAYER_H_

#include <map>
#include <string>
#include <thread>
#include <vector>

#include "proto/common.pb.h"
#include "proto/job.pb.h"
#include "utils/common.h"
#include "utils/blob.h"
#include "utils/param.h"

namespace singa {
using std::vector;
using std::string;

/**
 * Base layer class.
 *
 * Children should implement at least
 * Layer::ComputeFeature() and Layer::ComputGradient()
 * functions for contrastive-divergence/back-propagation algorithm.
 */
class Layer {
 public:
  static Layer* Create(const LayerProto& proto);

  Layer() {}
  virtual ~Layer() {}
  /**
   * Setup layer properties.
   *
   * Setup the shapes for data and parameters, also setup some properties
   * based on the layer configuration and connected layers.
   *
   * @param proto layer configuration.
   * @param npartitions num of total partitions of the original layer. This
   * layer should be setup as one partition.
   */
  virtual void Setup(const LayerProto& proto, int npartitions = 1) {
    CHECK_GE(npartitions, 1);
    layer_proto_ = proto;
  }
  /**
   * Compute features of this layer based on connected layers.
   *
   * @param perf pointer to Metric obj for collect and aggregate performance
   */
  virtual void ComputeFeature(int flag, Metric* perf) = 0;
  /**
   * Compute gradients for parameters and connected layers.
   * @param flag used to get the calling phase, e.g., forward of training
   * (kForward | kTrain)
   * @param flag used to get the calling phase, e.g., forward of training
   */
  virtual void ComputeGradient(int flag, Metric* perf) = 0;
  /**
   * Layers that have paramters must override this function.
   * @param flag used to get the calling phase, e.g., forward of training
   * (kForward | kTrain)
   * @return parameters associated with this layer
   */
  virtual const std::vector<Param*> GetParams() const {
    return std::vector<Param*> {};
  }
  /**
   * Return the connection type between one neuron of this layer and
   * its source layer.
   * Currently support two connection types: kOneToOne, and kOneToAll.
   * kOneToOne indicates the neuron depends on only one neuron from src layer.
   * kOneToAll indicates the neuron depends on all neurons from src layer.
   * TODO(wangwei) support kOneToMany.
   *
   * @param k index of source layer (current only support k = 0.
   * @param connection type.
   */
  virtual ConnectionType src_neuron_connection(int k) const {
    // CHECK_LT(k, srclayers_.size());
    return kOneToOne;
  }
  /**
   * Return the connection type of this layer and all dst layers.
   *
   * Currently support two connection types: kOneToOne, and kOneToMany.
   * kOneToOne indicates the users implement the ComputeFeature and
   * ComputeGradient function considering only one dest layer. In this case,
   * a SplitLayer will be added automatically to connect this layer with all
   * dest layer.
   * kOneToMany indicates the users has already considered multiple dest layers
   * in the implementation.
   * @return connection type default is kOneToOne.
   */
  virtual ConnectionType dst_layer_connection() const {
    return kOneToOne;
  }
  /**
   * For print debug info about each layer, e.g., norm of feature vector,
   * norm of parameters.
   *
   * @param step training/test/validation step
   * @param flag used to get the calling phase, e.g., forward of training
   * (kForward | kTrain)
   * @return debug info about this layer.
   */
  virtual const std::string DebugString(int step, int flag);
  /**
   * @return partition dimension of this layer.
   * -1 for no partition;
   *  0 for partition the mini-batch into sub-mini-batch.
   *  1 for partition the layer feature vector into sub-vector.
   */
  inline int partition_dim() const {
    CHECK_LE(layer_proto_.partition_dim(), 1);
    return layer_proto_.partition_dim();
  }
  inline int partition_id() const { return layer_proto_.partition_id(); }
  inline int type() const { return layer_proto_.type(); }
  /**
   * Return name of this layer
   */
  inline const std::string &name() const { return layer_proto_.name(); }
  /**
   * @return name of src data blob, used by prefetch layer to locate the data
   * blob in parser layers; The default value is "unknown"; If the
   * src layer is the prefetch layer and there are more than one parser layers,
   * this value be set.
  const std::string &datablob() const {
    return layer_proto_.datablob();
  }
   */
  /**
   * @return a const ref for Blob storing neuron values of this layer for BP
   */
  virtual const Blob<float>& data(const Layer* from) const {
    return data_;
  }
  virtual Blob<float>* mutable_data(const Layer* from) {
    return &data_;
  }
  virtual const Blob<float>& grad(const Layer* from) const {
    return grad_;
  }
  /**
   * @return a pointer to storing neuron grads of this layer for BP
   */
  virtual Blob<float>* mutable_grad(const Layer* from) {
    return &grad_;
  }
  /**
   * return LayerS that connected to this layer
   */
  inline const std::vector<Layer*> srclayers() const { return srclayers_; }
  /**
   * return LayerS that this layer connected to
   */
  inline const std::vector<Layer*> dstlayers() const { return dstlayers_; }
  inline int srclayers_size() const { return srclayers_.size(); }
  inline int dstlayers_size() const { return dstlayers_.size(); }
  inline void clear_dstlayers() { dstlayers_.clear(); }
  inline void clear_srclayers() { srclayers_.clear(); }
  inline void add_srclayer(Layer* src) { srclayers_.push_back(src); }
  inline void add_dstlayer(Layer* dst) { dstlayers_.push_back(dst); }

 protected:
  LayerProto layer_proto_;
  Blob<float> data_, grad_;
  std::vector<Layer*> srclayers_, dstlayers_;
};

/**
 * Base layer for connecting layers when neural net is partitioned.
 */
class ConnectionLayer : public Layer {
  // defined as a layer category
};

/**
 * Base layer for getting input data. May include layers for loading records,
 * parsing records.
 */
class InputLayer : public Layer {
  // defined as a layer category
};


class NeuronLayer : public Layer {
  // defined as a layer category
};

/**
 * Base layer for calculating loss and other metrics, e.g., precison.
 */
class LossLayer : public Layer {
 public:
  Blob<float>* mutable_grad(const Layer* layer) override {
    return nullptr;
  }
  const Blob<float>& grad(const Layer* from) const override {
    LOG(FATAL) << "Loss layer has no gradient blob";
    return grad_;
  }

 protected:
  Blob<float> metric_;
};

}  // namespace singa
#include "neuralnet/connection_layer.h"
#include "neuralnet/input_layer.h"
#include "neuralnet/loss_layer.h"
#include "neuralnet/neuron_layer.h"
#include "neuralnet/output_layer.h"

#endif  // SINGA_NEURALNET_BASE_LAYER_H_
