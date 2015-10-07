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

#ifndef SINGA_NEURALNET_INPUT_LAYER_PREFETCH_H_
#define SINGA_NEURALNET_INPUT_LAYER_PREFETCH_H_

#include <string>
#include <vector>
#include "singa/io/store.h"
#include "singa/neuralnet/layer.h"
#include "singa/utils/data_shard.h"
/**
 * \file this file includes the declarations of input layers that inherit the
 * base InputLayer to load input features.
 *
 * The feature loading phase can be implemented using a single layer or
 * separated into DataLayer (for loading features as records) and ParserLayer
 * (for parsing features from records). SINGA has provided some subclasses of
 * DataLayer and ParserLayer.
 *
 * Data prefetching can be implemented as a sub-class of InputLayer.
 * SINGA provides a built-in PrefetchLayer which embeds DataLayer and
 * ParserLayer.
 */
namespace singa {
using std::string;
using std::vector;
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
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override {}

 protected:
  std::thread thread_;
};

}  // namespace singa

#endif  // SINGA_NEURALNET_INPUT_LAYER_PREFETCH_H_
