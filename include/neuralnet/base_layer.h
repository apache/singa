#ifndef SINGA_NEURALNET_BASE_LAYER_H_
#define SINGA_NEURALNET_BASE_LAYER_H_

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
   * @param flag kTrain, kTest, kPositive, etc.
   */
  virtual void ComputeFeature(int flag, Metric* perf) = 0;
  /**
   * Compute gradients for parameters and connected layers.
   *
   * @param flag kTrain, kTest, kPositive, etc.
   */
  virtual void ComputeGradient(int flag, Metric* perf) = 0;
  virtual void ComputeGradient(int flag, Metric* perf) = 0;
  /**
   * For print debug info about each layer, e.g., norm of feature vector,
   * norm of parameters.
   *
   * @param step training/test/validation step
   * @param flag forward/backward/positive/negative...
   * @return debug info about this layer.
   */
  const string DebugString(int step, int flag);
  /**
   * Layers that have paramters must override this function.
   *
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
   * @param phase forward/backward/positive/negative...
   * @return debug info about this layer.
   */
  virtual const std::string DebugString(int step, Phase phase);
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
  virtual bool is_datalayer() const {
    return false;
  }
  virtual bool is_parserlayer() const {
    return false;
  }
  virtual bool is_losslayer() const {
    return false;
  }
  virtual bool is_bridgesrclayer() const {
    return false;
  }
  virtual bool is_bridgedstlayer() const {
    return false;
  }
  virtual bool is_bridgelayer() const {
    return false;
  }
  virtual bool is_vislayer() const {
    return false;
  }
  virtual bool is_hidlayer() const {
    return false;
  }

 protected:
  LayerProto layer_proto_;
<<<<<<< HEAD
  Blob<float> data_, grad_;
  vector<Layer*> srclayers_, dstlayers_;
};

class BridgeLayer : public Layer {
 public:
  void set_ready(bool a) {
    ready_ = a;
  }
  bool ready() const {
    return ready_;
  }
  bool is_bridgelayer() const override {
    return true;
  }

 protected:
  //!< true if received grad from BridgeDstLayer
  bool ready_;
};
/**
 * For sending data to layer on other threads which may resident on other nodes
 * due to layer/data partition.
 */
class BridgeSrcLayer: public BridgeLayer {
 public:
  using Layer::ComputeFeature;
  using Layer::ComputeGradient;

  void ComputeFeature(int flag, Metric* perf) override {}
  void ComputeGradient(int flag) override {
    ready_ = false;
  }

  const Blob<float>& data(const Layer* from) const override {
    return srclayers_[0]->data(this);
  }
  Blob<float>* mutable_data(const Layer* from) override {
    return srclayers_[0]->mutable_data(this);
  }
  const Blob<float>& grad(const Layer* from) const override {
    return srclayers_[0]->grad(this);
  }
  Blob<float>* mutable_grad(const Layer* from) override {
    return srclayers_[0]->mutable_grad(this);
  }

  bool is_bridgesrclayer() const override {
    return true;
  }
};
/**
 * For recv data from layer on other threads which may resident on other nodes
 * due to layer/data partiton
 */
class BridgeDstLayer: public BridgeLayer {
 public:
  using Layer::ComputeFeature;
  using Layer::ComputeGradient;

  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric* perf) override {
    // reset ready_ for next iteration.
    ready_ = false;
  }
  void ComputeGradient(int flag) override {}
  bool is_bridgedstlayer() const {
    return true;
  }
};

/**
 * Concate src layers on one dimension
 */
class ConcateLayer: public Layer {
 public:
  using Layer::ComputeFeature;
  using Layer::ComputeGradient;

  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric* perf) override;
  void ComputeGradient(int flag) override;
};

/**
 * Base layer for reading records from local Shard, HDFS, lmdb, etc.
 */
class DataLayer: public Layer {
 public:
  void ComputeGradient(int flag, Metric* perf) override {}
  bool is_datalayer() const override {
    return true;
  }
  Blob<float>* mutable_data(const Layer* layer) override {
    return nullptr;
  }
  Blob<float>* mutable_grad(const Layer* layer) override {
    return nullptr;
  }
  ConnectionType dst_layer_connection() const override {
    return kOneToMany;
  }
  inline int batchsize() const { return batchsize_; }
  virtual const Record& sample() const {
    return sample_;
  }
  /**
   * @return the loaded records
   */
  virtual const std::vector<Record>& records() const {
    return records_;
  }

 protected:
  int random_skip_;
  int batchsize_;
  Record sample_;
  std::vector<Record> records_;
};

/**
 * Base layer for parsing the input records into Blobs.
 */
class ParserLayer : public Layer {
 public:
  void ComputeFeature(Phase phase, Metric* perf) override;
  void ComputeGradient(Phase phase, Metric* perf) override {}
  /**
   * Parse records from DataLayer into blob.
   */
  virtual void ParseRecords(Phase phase, const std::vector<Record>& records,
      Blob<float>* blob) = 0;
  bool is_parserlayer() const override {
    return true;
  }
  Blob<float>* mutable_grad(const Layer* layer) override {
    return nullptr;
  }
  const Blob<float>& grad(const Layer* from) const  override {
    CHECK(false) << "Parser layer has not gradient blob";
    return grad_;
  }
};

class NeuronLayer : public Layer {
  // defined as a layer category
};

/**
 * Base layer for calculating loss and other metrics, e.g., precison.
 */
class LossLayer: public Layer {
 public:
  Blob<float>* mutable_grad(const Layer* layer) override {
    return nullptr;
  }
  const Blob<float>& grad(const Layer* from) const override {
    LOG(FATAL) << "Loss layer has no gradient blob";
    return grad_;
  }
  bool is_losslayer() const override {
    return true;
  }

 protected:
  Blob<float> metric_;
};

/**
 * Base layer for sending/waiting remote messages.
 */
class BridgeLayer : public Layer {
 public:
  inline void set_ready(bool a) { ready_ = a; }
  inline bool ready() const { return ready_; }
  bool is_bridgelayer() const override { return true; }

 protected:
  //!< true if received grad from BridgeDstLayer
  bool ready_;
};

/**
 * Base layer for connecting layers when neural net is partitioned.
 */
class ConnectionLayer : public Layer {
  // defined as a layer category
};

/**
 * Layer for prefetching data records and parsing them.
 *
 * The data loading and parsing work is done by internal DataLayer and
 * ParserLayer respectively. This layer controls the prefetching thread, i.e.,
 * creating and joining the prefetching thread.
 */
class PrefetchLayer : public Layer {
 public:
  ~PrefetchLayer();
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(Phase phase, Metric* perf) override;
  void ComputeGradient(Phase phase, Metric* perf) override {}
  const Blob<float>& data(const Layer* from, Phase phase) const override;
  void Prefetch(Phase phase);
  Blob<float>* mutable_data(const Layer* layer, Phase phase) override;
  Blob<float>* mutable_grad(const Layer* layer) override {
    return nullptr;
  }
  const Blob<float>& grad(const Layer* from) const override {
    CHECK(false) << "Loss layer has not gradient blob";
    return grad_;
  }

 protected:
  std::vector<Layer*> sublayers_;
  std::map<std::string, Blob<float>> datablobs_;
  std::thread thread_;
};

class RBMLayer: public Layer {
 public:
  const Blob<float>& neg_data(const Layer* layer) {
    return neg_data_;
  }
  Blob<float>* mutable_neg_data(const Layer* layer) {
    return &neg_data_;
  }
  const vector<Param*> GetParams() const override {
    vector<Param*> params{weight_, bias_};
    return params;
  }
  virtual Blob<float>* Sample(int flat) = 0;

 protected:
  //! dimension of the hidden layer
  int hdim_;
  //! dimension of the visible layer
  int vdim_;
  int batchsize_;
  Param* weight_, *bias_;

  Blob<float> neg_data_;
  Blob<float> neg_sample_;
  Blob<float> sample_;
};
}  // namespace singa

#endif  // SINGA_NEURALNET_BASE_LAYER_H_
