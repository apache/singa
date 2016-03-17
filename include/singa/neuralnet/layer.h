/************************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/

#ifndef SINGA_NEURALNET_LAYER_H_
#define SINGA_NEURALNET_LAYER_H_

#include <string>
#include <vector>
#include "singa/proto/common.pb.h"
#include "singa/proto/job.pb.h"
#include "singa/utils/common.h"
#include "singa/utils/blob.h"
#include "singa/utils/param.h"

namespace singa {
using std::vector;
using std::string;

// TODO(wangwei) make AuxType a template argument for Layer.
using AuxType = int;

inline const string AddUnrollingPrefix(int unroll_idx, const string& name) {
  return std::to_string(unroll_idx) + "#" + name;
}
inline const string AddPartitionSuffix(int partition_idx, const string& name) {
  return name + "@" + std::to_string(partition_idx);
}


inline const string AddPrefixSuffix(int unroll_idx, int partition_idx,
    const string& name) {
  return std::to_string(unroll_idx) + "#" + name + "@" +
    std::to_string(partition_idx);
}
/**
 * Base layer class.
 *
 * Subclasses should implement at least
 * Layer::ComputeFeature() and Layer::ComputGradient()
 * functions in accordance with the NeuralNet::TrainOneBatch function.
 */

class Layer {
 public:
  /**
   * Create a sub-layer instance based on proto.type();
   *
   * @param proto configuration of the layer instance.
   * @return pointer to the newly created layer instance.
   */
  static Layer* Create(const LayerProto& proto);

  Layer() {}
  virtual ~Layer() {}

  /**
   * Create for python binding, production test mode
   *
   */
  static Layer* CreateLayer(const string str);
  static void SetupLayer(Layer* layer, const string str, const vector<Layer*>& srclayers);

  /**
   * Setup layer properties.
   *
   * Setup members e.g., shapes of Param objects based on the layer
   * configuration and connected layers.
   * It should check the partition setting when setup the properties.
   *
   * @param conf layer configuration.
   * @param srclayers source layers that connect to this layer.
   */
  virtual void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) {
    layer_conf_ = conf;
    datavec_.push_back(&data_);
    gradvec_.push_back(&grad_);
  }


  /**
   * Compute features of this layer based on connected layers.
   *
   * @param[in] flag set by the TrainOneBatch function, e.g., to indicate the
   * running phase (kForward|kTrain, kForward|kTest, etc).
   * @param[in] srclayers source layers that connect to this layer.
   */
  virtual void ComputeFeature(int flag, const vector<Layer*>& srclayers) = 0;
  /**
   * Compute gradients for parameters associated with this layer.
   * It may also compute the gradients of the loss w.r.t the source layers.
   *
   * \copydetails ComputeFeature().
   */
  virtual void ComputeGradient(int flag, const vector<Layer*>& srclayers) = 0;
  /**
   * Layers that have paramters must override this function to return all Param
   * objects associated with this layer.
   *
   * @return parameters associated with this layer.
   */
  virtual const std::vector<Param*> GetParams() const {
    return std::vector<Param*> {};
  }
  virtual void SetParams(std::vector<Param*>) {}
  /**
   * Return the connection type between one neuron of this layer and its source
   * layer.
   *
   * Currently support two connection types: kOneToOne, and kOneToAll.
   * - kOneToOne indicates the neuron depends on only one neuron from src layer.
   * - kOneToAll indicates the neuron depends on all neurons from src layer.
   * TODO(wangwei) support kOneToMany.
   *
   * @param[in] k index of source layer, current only support k = 0.
   * @return connection type.
   */
  virtual ConnectionType src_neuron_connection(int k) const {
    // CHECK_LT(k, srclayers_.size());
    return kOneToOne;
  }
  /**
   * Return the connection type of this layer and all dst layers.
   *
   * Currently support two connection types: kOneToOne, and kOneToMany.
   * - kOneToOne indicates the users implement the ComputeFeature and
   * ComputeGradient function considering only one dst layer. In this case,
   * a SplitLayer will be added automatically to connect this layer with all
   * dest layer.
   * - kOneToMany indicates this layer has already considered multiple dst
   *   layers in the implementation.
   *
   * @return connection type default is kOneToOne.
   */
  virtual ConnectionType dst_layer_connection() const {
    return kOneToOne;
  }
  /**
   * To display layer info, e.g., aggreated loss/accuracy, or norm of feature
   * vector and norm of parameters.
   *
   * @param[in] debug whether print the debug info
   * @param[in] flag used to get the calling phase, e.g., forward of training
   * (kForward | kTrain).
   * @return info string about this layer, which is printed into the log.
   */
  virtual const std::string ToString(bool debug, int flag);
  /**
   * @return partition dimension of this layer,
   * - -1 for no partition.
   * -  0 for partition on the data dimension, i.e., partitioning the mini-batch
   *    into sub-mini-batches.
   * -  1 for partition this layer on feature dimension, i.e., the feature
   *    vector of each instance is partitioned into sub-vectors.
   */
  inline int partition_dim() const {
    CHECK_LE(layer_conf_.partition_dim(), 1);
    return layer_conf_.partition_dim();
  }
  /**
   * @return the partition ID (i.e., the worker ID to whom is layer is
   * dispatched) of this layer, which is a sublayer partitioned from the
   * original layer.
   */
  inline int partition_id() const { return layer_conf_.partition_id(); }
  /**
   * @return total number of partitions (i.e., sub-layers) of the original
   * layer of this layer.
   */
  inline int num_partitions() const { return layer_conf_.num_partitions(); }
  /**
   * @return the type of this layer, only valid for built-in layer (types).
   */
  inline LayerType type() const { return layer_conf_.type(); }
  /**
   * @return user-defined layer type.
   */
  inline const std::string& user_type() const {
    return layer_conf_.user_type();
  }
  /**
   * Return name of this layer
   */
  inline const std::string& name() const { return layer_conf_.name(); }
  /**
   * Return the index of the unrolled layer within the unrolling group, which
   * should be [0, max_unrolling_length)
   */
  inline const int unroll_index() const { return layer_conf_.unroll_index(); }

  /**
   * @return a const ref for Blob vector storing feature values of this layer.
   */
  virtual const vector<Blob<float>*>& data() const {
    return datavec_;
  }

  /**
   * @param[in] from pointer to one of the dst layer. For some layers, they have
   * more than one data Blob. In this case, this argument identifies the layer
   * that is requesting the data Blob.
   * @return a const ref for Blob storing feature values of this layer.
   * @deprecated {This function will be deleted, use
   * virtual const vector<Blob<float>>& data() const or
   * virtual const Blob<float>& data(int k) const instead}.
   */
  virtual const Blob<float>& data(const Layer* from) {
    return data_;
  }
  /**
   * @return a const ref for the kth Blob.
   * TODO(wangwei) if make this function const, there will be a warning
   * indicating that data(const Layer*) and this function are ambiguous for
   * data(0).
   */
  virtual const Blob<float>& data(int k) {
    return *datavec_.at(k);
  }

  /**
   * @see data().
   * @return the pointer to the Blob storing feature values of this layer.
   * @deprecated {This function will be deleted, use
   * virtual Blob<float>* mutable_data(int k) instead}.
   */
  virtual Blob<float>* mutable_data(const Layer* from) {
    return &data_;
  }
  /**
   * @return the pointer to the kth Blob.
   */
  virtual Blob<float>* mutable_data(int k) {
    return datavec_.at(k);
  }
  /**
   * @return auxiliary data, e.g., image label.
   */
  virtual const vector<AuxType>& aux_data(const Layer* from = nullptr) const {
    return aux_data_;
  }
  /**
   * @see data().
   * @return the const ref of the Blob for the gradient of this layer, mainly
   * used in BP algorithm.
   * @deprecated {This function will be deleted, use
   * virtual const vector<Blob<float>>& grad() const or
   * virtual const Blob<float>& grad(int k) const instead}.
   */
  virtual const Blob<float>& grad(const Layer* from) {
    return grad_;
  }
  /**
   * @see data().
   * @return the const ref of the Blob vector for the gradient of this layer.
   */
  virtual const vector<Blob<float>*>& grad() const {
    return gradvec_;
  }
  /**
   * @return the const ref of the kth Blob for the gradient of this layer.
   */
  virtual const Blob<float>& grad(int k) const {
    return *gradvec_.at(k);
  }
  /**
   * @see data().
   * @return a pointer to the Blob storing gradients of this layer, mainly
   * used in BP algorithm.
   */
  virtual Blob<float>* mutable_grad(const Layer* from) {
    return &grad_;
  }
  /**
   * @see data().
   * @return a pointer to the kth Blob storing gradients of this layer, mainly
   * used in BP algorithm.
   */
  virtual Blob<float>* mutable_grad(int k) {
    return gradvec_.at(k);
  }

 protected:
  LayerProto layer_conf_;
  Blob<float> data_, grad_;
  vector<AuxType> aux_data_;
  vector<Blob<float>*> datavec_, gradvec_;
};
/**************** Layer categories *****************/
/**
 * Base layer for connecting layers when neural net is partitioned.
 */
class ConnectionLayer : virtual public Layer {
  // defined as a layer category
};


/**
 * Base layer for getting input data. May include layers for loading records,
 * parsing records.
 */
class InputLayer : virtual public Layer {
 public:
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override {}
  ConnectionType dst_layer_connection() const override { return kOneToMany; }
  Blob<float>* mutable_grad(const Layer* layer) override {
    return nullptr;
    // LOG(FATAL) << "Input layer has no gradient blob";
  }
  const Blob<float>& grad(const Layer* from) override {
    return grad_;
    // LOG(FATAL) << "Input layer has no gradient blob";
  }
};

using SingleLabelImageRecord = RecordProto;

/**
 * Base layer for feature transformation, e.g., ConvolutionLayer, PoolingLayer,
 * etc.
 */
class NeuronLayer : virtual public Layer {
  // defined as a layer category
};


/**
 * Base layer for calculating loss and doing BackPropagation.
 */
class LossLayer : virtual public Layer {
 public:
  Blob<float>* mutable_grad(const Layer* layer) override {
    return nullptr;
    // LOG(FATAL) << "Loss layer has no gradient blob";
  }
  const Blob<float>& grad(const Layer* from) override {
    return grad_;
    // LOG(FATAL) << "Loss layer has no gradient blob";
  }
};

/**
 * Base layer for collecting features into disk file, HTTP stream, etc.
 */
class OutputLayer : virtual public Layer {
 public:
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override {}
  Blob<float>* mutable_grad(const Layer* layer) override {
    return nullptr;
    // LOG(FATAL) << "Output layer has no gradient blob";
  }
  const Blob<float>& grad(const Layer* from) override {
    return grad_;
    // LOG(FATAL) << "Output layer has no gradient blob";
  }
};


}  // namespace singa
#endif  // SINGA_NEURALNET_LAYER_H_
