#ifndef SINGA_NEURALNET_BASE_LAYER_H_
#define SINGA_NEURALNET_BASE_LAYER_H_

#include <vector>
#include <string>
#include <map>
#include <utility>
#include <memory>
#include <thread>

#include "proto/job.pb.h"
#include "proto/common.pb.h"
#include "utils/param.h"
#include "utils/common.h"
#include "utils/blob.h"

namespace singa {

using std::vector;
using std::string;
using std::map;


class Layer;
/**
 * Base layer class.
 *
 * Children should implement at least
 * Layer::ComputeFeature() and Layer::ComputGradient()
 * functions for contrastive-divergence/back-propagation algorithm.
 */
class Layer {
 public:
  Layer() { }
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
  virtual void Setup(const LayerProto& proto, int npartitions = 1);

  /**
   * Compute features of this layer based on connected layers.
   *
   * @param phase kTrain, kTest, kPositive, etc.
   */
  virtual void ComputeFeature(Phase phase, Metric* perf) = 0;
  /**
   * Compute gradients for parameters and connected layers.
   *
   * @param phase kTrain, kTest, kPositive, etc.
   */
  virtual void ComputeGradient(Phase phase) = 0;

  /**
   * For print debug info about each layer, e.g., norm of feature vector,
   * norm of parameters.
   *
   * @param step training/test/validation step
   * @param phase forward/backward/positive/negative...
   * @return debug info about this layer.
   */
  const string DebugString(int step, Phase phase);
  /**
   * Layers that have paramters must override this function.
   *
   * @return parameters associated with this layer
   */
  virtual const vector<Param*> GetParams() const {
    return vector<Param*> {};
  }
  /**
   * Return the connection type between one neuron of this layer and
   * its source layer.
   * Currently support two connection types: kOneToOne, and kOneToAll.
   * kOneToOne indicates the neuron depends on only one neuron from src layer.
   * kOneToAll indicates the neuron depends on all neurons from src layer.
   * TODO support kOneToMany.
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
   * @return partition dimension of this layer.
   * -1 for no partition;
   *  0 for partition the mini-batch into sub-mini-batch.
   *  1 for partition the layer feature vector into sub-vector.
   */
  virtual int partition_dim() const {
    return layer_proto_.partition_dim();
  }

  virtual int partition_id() const {
    return layer_proto_.partition_id();
  }
  virtual int type() const {
    return layer_proto_.type();
  }
  /**
   * Return name of this layer
   */
  const std::string &name() const {
    return layer_proto_.name();
  }
  /**
   * @return name of src data blob, used by prefetch layer to locate the data
   * blob in parser layers; The default value is "unknown"; If the
   * src layer is the prefetch layer and there are more than one parser layers,
   * this value be set.
   */
  const std::string &datablob() const {
    return layer_proto_.datablob();
  }
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
  virtual const vector<Layer*> srclayers() const {
    return srclayers_;
  }
  /**
   * return LayerS that this layer connected to
   */
  virtual const vector<Layer*> dstlayers() const {
    return dstlayers_;
  }

  virtual int srclayers_size() const {
    return srclayers_.size();
  }
  virtual int dstlayers_size() const {
    return dstlayers_.size();
  }
  virtual void clear_dstlayers() {
    dstlayers_.clear();
  }
  virtual void clear_srclayers() {
    srclayers_.clear();
  }

  virtual void add_srclayer(Layer* src) {
    srclayers_.push_back(src);
  }
  virtual void add_dstlayer(Layer* dst) {
    dstlayers_.push_back(dst);
  }

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

 protected:
  LayerProto layer_proto_;
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

  void ComputeFeature(Phase phase, Metric* perf) override {}
  void ComputeGradient(Phase phase) override {
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
  void ComputeFeature(Phase phase, Metric* perf) override {
    // reset ready_ for next iteration.
    ready_ = false;
  }
  void ComputeGradient(Phase phase) override {}
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
  void ComputeFeature(Phase phase, Metric* perf) override;
  void ComputeGradient(Phase phase) override;
};

/**
 * Base layer for reading records from local Shard, HDFS, lmdb, etc.
 */
class DataLayer: public Layer{
 public:
  using Layer::ComputeGradient;
  using Layer::mutable_data;
  using Layer::mutable_grad;
  using Layer::dst_layer_connection;

  void ComputeGradient(Phase phase) override {}
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

  int batchsize() const {
    return batchsize_;
  }
  virtual const Record& sample() const {
    return sample_;
  }
  /**
   * @return the loaded records
   */
  virtual const vector<Record>& records() const {
    return records_;
  }

 protected:
  int random_skip_, batchsize_;
  Record sample_;
  vector<Record> records_;
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
  using Layer::ComputeFeature;
  using Layer::ComputeGradient;

  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(Phase phase, Metric* perf) override;
  void ComputeGradient(Phase phase) override {};

  const Blob<float>& data(const Layer* from) const override;
  Blob<float>* mutable_data(const Layer* layer) override;

  Blob<float>* mutable_grad(const Layer* layer) override {
    return nullptr;
  }
  const Blob<float>& grad(const Layer* from) const override {
    CHECK(false) << "Loss layer has not gradient blob";
    return grad_;
  }

  void Prefetch(Phase phase);
  virtual ~PrefetchLayer();

 protected:
  vector<Layer*> sublayers_;
  map<string, Blob<float>> datablobs_;
  std::thread thread_;
};

/**
 * Slice the source layer into multiple dst layers on one dimension
 */
class SliceLayer: public Layer {
 public:
  using Layer::ComputeFeature;
  using Layer::ComputeGradient;

  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(Phase phase, Metric* perf) override;
  void ComputeGradient(Phase phase) override;
  ConnectionType dst_layer_connection() const override {
    return kOneToMany;
  }
  const Blob<float>& data(const Layer* layer) const override;
  const Blob<float>& grad(const Layer* layer) const override;
  Blob<float>* mutable_data(const Layer* layer) override;
  Blob<float>* mutable_grad(const Layer* layer) override;

 protected:
  int SliceID(const Layer* layer) const;

 private:
  vector<Blob<float>> datavec_, gradvec_;
  int slice_dim_, slice_num_;
};

/**
 * Connect the source layer with multiple dst layers.
 * Pass source layer's data blob directly to dst layers.
 * Aggregate dst layer's gradients into source layer's gradient.
 */
class SplitLayer: public Layer {
 public:
  using Layer::ComputeFeature;
  using Layer::ComputeGradient;

  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(Phase phase, Metric* perf) override;
  void ComputeGradient(Phase phase) override;
  ConnectionType dst_layer_connection() const override {
    return kOneToMany;
  }
 protected:
  Blob<float> grads_;
};

/**
 * Loss layer to calculate loss and other metrics, e.g., precison.
 */
class LossLayer: public Layer{
 public:
  using Layer::mutable_grad;
  using Layer::grad;
  using Layer::is_losslayer;

  Blob<float>* mutable_grad(const Layer* layer) override {
    return nullptr;
  }
  const Blob<float>& grad(const Layer* from) const override {
    CHECK(false) << "Loss layer has not gradient blob";
    return grad_;
  }
  bool is_losslayer() const override {
    return true;
  }

 protected:
  Blob<float> metric_;
};

/**
 * parse the input records into Blobs.
 */
class ParserLayer: public Layer {
 public:
  using Layer::ComputeFeature;
  using Layer::ComputeGradient;
  using Layer::is_parserlayer;
  using Layer::mutable_grad;
  using Layer::grad;

  void ComputeFeature(Phase phase, Metric* perf) override;
  void ComputeGradient(Phase phase) override {};
  /**
   * Parse records from DataLayer into blob.
   */
  virtual void ParseRecords(Phase phase, const vector<Record>& records,
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
}  // namespace singa

#endif  // SINGA_NEURALNET_BASE_LAYER_H_
