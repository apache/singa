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

#ifndef SINGA_NEURALNET_INPUT_LAYER_STORE_INPUT_H_
#define SINGA_NEURALNET_INPUT_LAYER_STORE_INPUT_H_

#include <string>
#include <vector>
#include "singa/io/store.h"
#include "singa/neuralnet/layer.h"
namespace singa {
using std::string;
using std::vector;

/**
 * Base class for loading data from Store.
 */
class StoreInputLayer : virtual public InputLayer {
 public:
  ~StoreInputLayer();
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;

  ConnectionType dst_layer_connection() const override { return kOneToMany; }

 protected:
  /**
   * Parsing the (key, val) tuple to get feature (and label).
   * Subclasses must implment this function.
   * @param[in] k parse this tuple as the k-th instance of one mini-batch.
   * @param[in] flag used to guide the parsing, e.g., kDeploy phase should not
   * parse labels from the tuple.
   * @param[in] key
   * @param[in] val
   */
  virtual bool Parse(int k, int flag, const string& key, const string& val) = 0;

 protected:
  int batchsize_ = 1;
  int random_skip_ = 0;
  io::Store* store_ = nullptr;
};

/**
 * Base layer for parsing a key-value tuple as a feature vector with fixed
 * length. The feature shape is indicated by users in the configuration.
 * Each tuple may has a label.
 */
class SingleLabelRecordLayer : public StoreInputLayer {
 public:
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;

 protected:
  /**
   * Load a single record (tuple), e.g., the mean or standard variance vector.
   */
  virtual void LoadRecord(const string& backend, const string& path,
      Blob<float>* to) = 0;

 protected:
  /**
   * Feature standardization by processing each feature dimension via
   * @f$ y = (x - mu)/ std @f$
   * <a href= "http://ufldl.stanford.edu/wiki/index.php/Data_Preprocessing">
   * UFLDL</a>
   */
  Blob<float> mean_, std_;
};

}  // namespace singa

#endif  // SINGA_NEURALNET_INPUT_LAYER_STORE_INPUT_H_
