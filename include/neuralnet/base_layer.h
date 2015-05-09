#ifndef INCLUDE_BASE_LAYER_H_
#define INCLUDE_BASE_LAYER_H_

#include <vector>
#include <string>
#include <map>
#include <functional>
#include <utility>
#include <memory>
#include <chrono>
#include <algorithm>
#include <thread>

#include "proto/model.pb.h"
#include "utils/param.h"
#include "utils/common.h"
#include "utils/blob.h"

using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::string;
using std::map;

namespace singa{

class Layer;
typedef shared_ptr<Layer> SLayer;
/**
 * Base layer class.
 * Children should implement at least Layer::Setup, Layer::ComputeFeature(),
 * Layer::ComputGradient() functions for backpropagation method;
 * TODO(zhaojing) subclass the base layer class to support contrastive divergence,
 * The identifier of each layer is the literal string of the class name without
 * the suffix "Layer", which is used in layer registration and creation.
 */
class Layer {
 public:
  Layer(){}
  virtual ~Layer(){}
  /**
   * Layer initialization.
   *
   * It simply saves the proto configuation, most initializations are done by
   * Setup().
   *
   * @param proto user defined layer configuration
   */
  virtual void Init(const LayerProto &proto);
  /**
   * Copy layer configuration from the other Layer, and use the shape argument
   * to as its data shape.
   */
  void Init(const Layer& other, const vector<int>& shape);
  /**
   * TODO(wangsheng) Marshal layer properties and data into google protobuf
   * object (i.e., snapshot).
   *
   * Parameters are marshalled separately into another object (i.e., model).
   *
   * @param layer_proto
   * @param copyData if true marshal layer data, e.g., feature value
   */
  virtual void ToProto(LayerProto *layer_proto, bool copyData);
  /**
   * Setup layer properties.
   *
   * Setup the shapes for data and parameters, also setup some properties
   * based on the layer configuration and connected src layers.
   *
   * @param srclayers layers connecting to this layer
   */
  virtual void Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers)=0;
  /**
   * \copydoc Setup(const LayerProto&, const vector<SLayer>&)
   */
  virtual void Setup();
  /**
   * Setup the layer properties except shape.
   *
   * The shape is already set and passed in to set other properties.
   * properties are set according to shapes of itself and connected layers, and
   * configuration. this should not change the current shape_(
   * shape check is done outside the function).
   */
  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers)=0;
  /**
   * \copybrief SetupAfterPartition(const LayerProto&, const vector<int> &,
   * const vector<SLayer>& ).
   */
  virtual void SetupAfterPartition();
  /**
   * Layers that have paramters must overload this function.
   *
   * @return parameters associated with this layer
   */
  virtual vector<shared_ptr<Param>> GetParams(){
    return vector<shared_ptr<Param>>();
  }
  /**
   * Compute features of this layer based on connected layers.
   *
   * Implement forward propagation for BP.
   * TODO(zhaojing) Implement both postive phase and negative phase for CD.
   *
   * @param training true if in training phase
   * @param srclayers layers connecting to this layer
   */
  virtual void ComputeFeature(bool training, const vector<SLayer>& srclayers)=0;
  /**
   * \copybrief ComputeFeature(const vector<SLayer>& srclayers)
   */
  virtual void ComputeFeature(bool training);
  /**
   * Compute gradients for parameters and connecting layers.
   *
   * Implement backward propagation for BP.
   * TODO(zhaojing) Calculate gradients for parameters for CD.
   *
   * @param srclayers layers connecting to this layer.
   */
  virtual void ComputeGradient(const vector<SLayer>& srclayers)=0;
  /**
   * \copybrief ComputeGradient(const vector<SLayer>& srclayers)
   */
  virtual void ComputeGradient();
  /**
   * Decide on which dimension to do the partitioning.
   *
   * @mode kLayer, kData, kNone (no partition)
   * @return the partition dimension, -1 for no partition
   */
  virtual int partition_dimension() const {
    int ret=0;
    if(partition_type()==kLayerPartition)
      ret= 1;
    else if(partition_type()==kNone)
      ret= -1;
    return ret;
  }

  /**
   * Return connection type between two layers.
   *
   * Currently support two connections: kOneToOne, and kOneToAll.
   * kOneToOne indicates the dst neuron depends on only one neuron from src
   * layer. kOneToAll indicates the dst neuron depends on all neurons from src
   * layer. TODO support kOneToMany.
   */
  virtual ConnectionType connection_type(int k) const {
    CHECK_LT(k, srclayers_.size());
    return kOneToOne;
  }
  /**
   * @return partition type of this layer, e.g., kNone, kLayer or kData.
   */
  virtual PartitionType partition_type() const {
    return layer_proto_.partition_type();
  }
  /**
   * Set location ID as the worker ID within a worker group.
   * TODO(wangwei) merge location ID with partition ID
   */
  virtual void set_locationid(int id){
    layer_proto_.set_locationid(id);
  }
  /**
   * @return location ID
   */
  virtual int locationid() const {
    return layer_proto_.locationid();
  }
  /**
   * partition id is the ID of the layer in the original layer.
   */
  virtual void set_partitionid(int id){
    layer_proto_.set_partitionid(id);
  }
  virtual int partitiionid() const {
    return layer_proto_.partitionid();
  }
  virtual void set_name(string name){
    name_=name;
    layer_proto_.set_name(name);
  }
  virtual const string type() const {
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
   * this value value be set.
   */
  const std::string &datablob() const {
    return layer_proto_.datablob();
  }
  const vector<int>& shape(const Layer* layer) const{
    return data(layer).shape();
  }

  /**
   * @return a const ref for Blob storing neuron values of this layer for BP
   */
  virtual const Blob<float>& data(const Layer* from) const {
    return data_;
  }
  virtual Blob<float>* mutable_data(const Layer* from){
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
  virtual const vector< SLayer> srclayers() const {
    return srclayers_;
  }
  /**
   * return LayerS that this layer connected to
   */
  virtual const vector<SLayer> dstlayers() const {
    return dstlayers_;
  }

  virtual const int srclayers_size() const {
    return srclayers_.size();
  }
  virtual const int dstlayers_size() const {
    return dstlayers_.size();
  }
  virtual void ClearDstLayers() {
    dstlayers_.clear();
  }
  virtual void ClearSrcLayers() {
    srclayers_.clear();
  }

  virtual void AddSrcLayer(SLayer src){
    srclayers_.push_back(src);
  }
  virtual void AddDstLayer(SLayer dst){
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
protected:
  string name_;
  Blob<float> data_, grad_;
  LayerProto layer_proto_;
  vector<SLayer> srclayers_, dstlayers_;
};

/**
 * For sending data to layer on other threads which may resident on other nodes
 * due to layer/data partition.
 */
class BridgeSrcLayer: public Layer {
 public:
  virtual void Setup(const LayerProto& proto, const vector<SLayer>& srclayers);
  virtual void SetupAfterPartition();
  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){}

  virtual void ComputeFeature(bool training, const vector<SLayer>& srclayers);
  virtual void ComputeGradient(const vector<SLayer>& srclayers);
  virtual bool is_bridgesrclayer() const {
    return true;
  }

  virtual void set_ready(bool a) {
    ready_=a;
  }
  virtual bool ready() const {
    return ready_;
  }
 protected:
  bool ready_;
};
/**
 * For recv data from layer on other threads which may resident on other nodes
 * due to layer/data partiton
 */
class BridgeDstLayer: public Layer {
 public:
  virtual void Setup(const LayerProto& proto, const vector<SLayer>& srclayers);
  virtual void SetupAfterPartition();
  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){}

  virtual void ComputeFeature(bool training, const vector<SLayer>& srclayers);
  virtual void ComputeGradient(const vector<SLayer>& srclayers);
  virtual bool is_bridgedstlayer() const {
    return true;
  }
  virtual void set_ready(bool a) {
    ready_=a;
  }
  virtual bool ready() const {
    return ready_;
  }
 protected:
  bool ready_;
};

/**
 * Concate src layers on one dimension
 */
class ConcateLayer: public Layer {
 public:
  virtual void Setup(const LayerProto& proto, const vector<SLayer>& srclayers);
  virtual void SetupAfterPartition();
  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){}

  virtual void ComputeFeature(bool training, const vector<shared_ptr<Layer>>& srclayers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& srclayers);
};


/**
 * Base layer for reading records from local Shard, HDFS, lmdb, etc.
 * Cannot be partitioned, always returns kNone for partition type.
 */

class DataLayer: public Layer{
 public:
  virtual void ComputeFeature(bool training, const vector<SLayer>& srclayers)=0;
  virtual void Setup(const LayerProto& proto, const vector<SLayer>& srclayers)=0;
  virtual bool is_datalayer() const {
    return true;
  }
  virtual void ComputeGradient(const vector<SLayer>& srclayers){};
  virtual const vector<Record>& records() const {
    return records_;
  }
  virtual void Setup(){
    vector<SLayer> dummy;
    Setup(layer_proto_,dummy);
    has_setup_=true;
  }
  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){}

  virtual void SetupAfterPartition(){
    if(!has_setup_)
    Setup();
  }
  virtual PartitionType partition_type () const {
    return kNone;
  }

  virtual int batchsize() const {
    return layer_proto_.data_param().batchsize();
  }
  virtual const Record& sample() const {
    return sample_;
  }

  virtual Blob<float>* mutable_data(const Layer* layer) {
    return nullptr;
  }
  virtual Blob<float>* mutable_grad(const Layer* layer) {
    return nullptr;
  }
 protected:
  bool has_setup_;
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
  virtual ~PrefetchLayer();
  virtual void ComputeFeature(bool training, const vector<SLayer>& srclayers);
  virtual void Setup(const LayerProto& proto, const vector<SLayer>& srclayers);
  virtual const Blob<float>& data(const Layer* from) const ;
  virtual Blob<float>* mutable_data(const Layer* layer) ;
  virtual void ComputeGradient(const vector<SLayer>& srclayers){};
  virtual Blob<float>* mutable_grad(const Layer* layer){
    return nullptr;
  }
  virtual const Blob<float>& grad(const Layer* from) const {
    CHECK(false)<<"Loss layer has not gradient blob";
    return grad_;
  }

  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){}

  virtual PartitionType partition_type () const {
    return kNone;
  }

  void Prefetch(bool training);
 protected:
  vector<shared_ptr<Layer>> sublayers_;
  map<string, Blob<float>> datablobs_;
  std::thread thread_;
};

/**
 * Slice the source layer into multiple dst layers on one dimension
 */
class SliceLayer: public Layer {
 public:
  virtual void Setup(const LayerProto& proto, const vector<SLayer>& srclayers);
  virtual void SetupAfterPartition();
  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){}


  virtual const Blob<float>& data(const Layer* layer) const;
  virtual const Blob<float>& grad(const Layer* layer) const;
  virtual Blob<float>* mutable_data(const Layer* layer);
  virtual Blob<float>* mutable_grad(const Layer* layer);
  virtual void ComputeFeature(bool training, const vector<shared_ptr<Layer>>& srclayers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& srclayers);

 protected:
  int SliceID(const Layer* layer) const;
  vector<Blob<float>> datavec_, gradvec_;
};

/**
 * Replciate this layer into multiple dst layers
 * TODO change name to ReplicateLayer.
 */
class SplitLayer: public Layer {
 public:
  virtual void Setup(const LayerProto& proto, const vector<SLayer>& srclayers);
  virtual void SetupAfterPartition();
  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){}

  virtual void ComputeFeature(bool training, const vector<shared_ptr<Layer>>& srclayers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& srclayers);
};

/**
 * Loss layer to calculate loss and other metrics, e.g., precison.
 */
class LossLayer: public Layer{
 public:
  virtual void Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers)=0;

  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers)=0;
  virtual Blob<float>* mutable_grad(const Layer* layer){
    return nullptr;
  }
  virtual const Blob<float>& grad(const Layer* from) const {
    CHECK(false)<<"Loss layer has not gradient blob";
    return grad_;
  }
  virtual bool is_losslayer() const {
    return true;
  }

  virtual const Blob<float>& metric() const {
    return metric_;
  }
 protected:
  Blob<float> metric_;
};

/**
 * parse the input records into Blobs.
 */
class ParserLayer: public Layer {
 public:
  virtual void Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers)=0;
  /**
   * Parse records from DataLayer into blob.
   * This function is called by
   * ComputeFeature(bool, const vector<SLayer>& srclayers)  or Prefetch(bool).
   */
  virtual void ParseRecords(bool training, const vector<Record>& records,
      Blob<float>* blob)=0;

  virtual bool is_parserlayer() const {
    return true;
  }

  virtual void ComputeFeature(bool training, const vector<SLayer>& srclayers);
  /**
   * Dummy function. ParserLayer does not compute gradients.
   */
  virtual void ComputeGradient(const vector<SLayer>& srclayers){};
  virtual void Setup(){
    Setup(layer_proto_,srclayers_);
    has_setup_=true;
  }
  virtual void SetupAfterPartition(){
    if(!has_setup_)
      Setup();
  }

  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers){}

  virtual PartitionType partition_type () const{
    return kNone;
  }
  virtual Blob<float>* mutable_grad(const Layer* layer) {
    return nullptr;
  }
  virtual const Blob<float>& grad(const Layer* from) const {
    CHECK(false)<<"Parser layer has not gradient blob";
    return grad_;
  }

 private:
  bool has_setup_;
};
} // singa

#endif // INCLUDE_BASE_LAYER_H_
