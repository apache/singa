/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef SINGA_MODEL_FEED_FORWARD_NET_H_
#define SINGA_MODEL_FEED_FORWARD_NET_H_
#include <memory>
#include <thread>

#include "singa/model/layer.h"
#include "singa/model/loss.h"
#include "singa/model/metric.h"
#include "singa/model/updater.h"
namespace singa {

/// The feed-forward neural net.
/// It provides functions for constructing the layers, access layer parameters,
/// and conducting training, evaluation and prediction.
class FeedForwardNet {
 public:
  FeedForwardNet() = default;
  /// Delete all layers.
  ~FeedForwardNet();

  /// Add a layer with the assumption that
  /// 1. this function is called in correct order, i.e., the layers are added
  ///    following the topological order.
  /// 2. this layer has already been setup (Setup function is called outside).
  /// The layer will be freed in the destructor of FeedForwardNet.
  std::shared_ptr<Layer> Add(std::shared_ptr<Layer> layer);

  // TODO(wangwei) add ConcatenateLayer and SliceLayer
  // AddConcatenateLayer(vector<Layer*> src, Layer *dst);
  // AddSliceLayer(Layer* layer, vector<Layer*> dst);

  /// Add a layer by providing its configuration, and setup it.
  /// Assume the layer is added in corret order.
  /// For the first layer, 'sample_shape' (the input sample shape) is necessary
  /// for calling Setup().
  std::shared_ptr<Layer> Add(const LayerConf& conf,
                             const Shape* sample_shape = nullptr);

  /// Set some fields used for training and evaluating the neural net.
  /// This method will instantiate an Updater ,then wrap the Optimier into
  /// Updater and always register the parameters of the net instance.
  /// If the neural net is constructed for evaluation only, then 'opt' is not
  /// necessary; But for training, both 'opt' and 'loss' are necessary.
  /// 'shuffle' indicates shuffling training samples within one epoch it is
  /// valid using Train(). If to_register is set true, parameter will be
  /// registered in Updater.;
  void Compile(bool shuffle, Optimizer* opt, Loss* loss, Metric* metric);
  /// Set some fields used for training and evaluating the neural net.
  /// This method is mainly used in parallel training, where we need
  /// multiple neuralnet instances.
  /// If the neural net is constructed for evaluation only, then 'updater' is
  /// not
  /// necessary; But for training, both 'opt' and 'loss' are necessary.
  /// 'shuffle' indicates shuffling training samples within one epoch it is
  /// valid using Train(). If to_register is set true, parameter will be
  /// registered in Updater.;
  void Compile(bool shuffle, bool to_register, std::shared_ptr<Updater> updater,
               Loss* loss, Metric* metric);

  /// Conduct the training giving the training data 'x' and label 'y'.
  /// 'val_split' of training data is used for
  /// validation. Validation is performance before every epoch.
  /// Due to memory limit, 'x' and 'y' could not be very large. Hence, it is
  /// typically used for small training datasets, e.g., cifar10 and MNIST which
  /// can be stored in main memory.
  void Train(size_t batchsize, int nb_epoch, const Tensor& x, const Tensor& y,
             float val_split = 0.0f);
  /// Conduct the training given the training and validation data.
  /// Validation is performance before every epoch.
  /// Due to memory limit, 'x' and 'y' could not be very large. Hence, it is
  /// typically used for small training datasets, e.g., cifar10 and MNIST which
  /// can be stored in main memory.
  void Train(size_t batchsize, int nb_epoch, const Tensor& x, const Tensor& y,
             const Tensor& val_x, const Tensor& val_y);
  /// Train the neural net over one batch of training data.
  const std::pair<float, float> TrainOnBatch(int epoch, const Tensor& x,
                                             const Tensor& y);

  /// Evaluate the neural net with given data.
  /// Returns one tensor for loss values and one tensor for metric values;
  /// Each sample would have a loss value and a metric value (if 'metic' is set
  /// in Compile()).'batchsize' is used for controlling the memory footprint.
  /// It should be smaller than the total number of samples.
  /// Due to memory limit, 'x' and 'y' could not be very large. Hence, it is
  /// typically used for small training datasets, e.g., cifar10 and MNIST which
  /// can be stored in main memory.
  std::pair<Tensor, Tensor> Evaluate(const Tensor& x, const Tensor& y,
                                     size_t batchsize = 128);
  /// Evaluate the neural net for one batch of data
  std::pair<Tensor, Tensor> EvaluateOnBatch(const Tensor& x, const Tensor& y);

  /// Predict the probability distributation over candicate classes for each
  /// data sample. 'batchsize' is used for controlling the memory footprint.
  /// It should be smaller than the total number of samples.
  /// Due to memory limit, 'x' and 'y' could not be very large. Hence, it is
  /// typically used for small training datasets, e.g., cifar10 and MNIST which
  /// can be stored in main memory.
  const Tensor Predict(const Tensor& x, size_t batchsize = 128);
  /// Predict for one batch data.
  const Tensor PredictOnBatch(const Tensor& x);

  /// Forward layers one by one using the data batch 'x'.
  /// Returns the prediction results (from the last layer).
  const Tensor Forward(int flag, const Tensor& x);
  /// Backward layers one by one using the gradient batch 'grad'.
  /// Returns the parameter gradients.
  const vector<Tensor> Backward(int flag, const Tensor& grad);

  /// Clone the neuaral net by cloning every layer to the given device.
  /// If 'device' is nullptr, then clone it one the current device.
  FeedForwardNet Clone(std::shared_ptr<Device> device);
  /// Move the layer data to the given device.
  void ToDevice(std::shared_ptr<Device> device);
  void ToHost() { ToDevice(defaultDevice); }
  /// Set the data type of each layer.
  void AsType(DataType dtype);

  /// A wrapper method to spawn a thread to execute Train() method.
  std::thread TrainThread(size_t batchsize, int nb_epoch, const Tensor& x,
                          const Tensor& y, const Tensor& val_x,
                          const Tensor& val_y) {
    return std::thread(
        [=]() { Train(batchsize, nb_epoch, x, y, val_x, val_y); });
  }

  /// A wrapper method to spawn a thread to execute Train() method.
  std::thread TrainThread(size_t batchsize, int nb_epoch, const Tensor& x,
                          const Tensor& y) {
    return std::thread([=]() { Train(batchsize, nb_epoch, x, y); });
  }

  const vector<std::shared_ptr<Layer>> layers() const { return layers_; }
  const vector<string> GetParamNames() const;
  const vector<ParamSpec> GetParamSpecs() const;
  const vector<Tensor> GetParamValues() const;

 protected:
  vector<std::shared_ptr<Layer>> layers_;
  std::shared_ptr<Updater> updater_;
  Loss* loss_;
  Metric* metric_;

  bool shuffle_ = true;
  Device* device_ = nullptr;
  DataType dtype_ = kFloat32;
};

}  // namespace singa

#endif  // SINGA_MODEL_FEED_FORWARD_NET_H_
