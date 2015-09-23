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

#ifndef SINGA_NEURALNET_NEURALNET_H_
#define SINGA_NEURALNET_NEURALNET_H_

#include <string>
#include <vector>

#include "neuralnet/layer.h"
#include "proto/job.pb.h"
#include "utils/factory.h"
#include "utils/graph.h"

namespace singa {

/**
 * The neural network is constructed from user configurations in NetProto.
 *
 * Some layers, e.g., SplitLayer and BridgeSrcLayer/BridgeDstLayer
 * will be added implicitly to partition the neural network.
 * TODO(wangwei) create wrappers for popular models, e.g., MLP, CNN.
 */
class NeuralNet {
 public:
  /**
   * Create the neural network for training, test or validation.
   *
   * Parameters for test/validation net can share those from training after
   * setup (done outside of this funcion).
   *
   * @param net_conf proto for the neural network
   * @param phase test/training/validation
   * @param npartitions num of partitions, do partitioning if num > 1
   * @return pointer to a neural net
   */
  static NeuralNet* Create(const NetProto& net_conf, Phase phase,
                           int npartitions);

  /**
   * construct the net structure from protocol buffer.
   * @param netproto neural net config
   * @param npartitions num of partitions. 1 for no partitioning.
   */
  NeuralNet(NetProto netproto, int npartitions);
  ~NeuralNet();
  /**
   * To display the adjacency layers
   */
  std::string ToAdjacency();
  /**
   * Share memory of parameter values from other neuralnet
   */
  void ShareParamsFrom(NeuralNet* other);
  inline const std::vector<Layer*>& layers() { return layers_; }
  inline const std::vector<Param*>& params() const { return params_; }
  inline Layer* name2layer(std::string name) const {
    if (name2layer_.find(name) != name2layer_.end())
      return name2layer_.at(name);
    else
      return nullptr;
  }
  inline Param* paramid2param(int id) const { return paramid2param_.at(id); }

 protected:
  /**
   * Create a neural net graph, one node for each layer.
   *
   * Partition the graph if npartitions > 1, each layer is sliced according to
   * its own partition setting.
   * @param netproto
   * @npartitions
   * @return neural net graph
   */
  Graph* CreateGraph(const NetProto& netproto, int npartitions);
  /**
   * Create neural net from graph, one layer per node.
   */
  void CreateNetFromGraph(Graph* graph, int npartitions);
  /**
   * prepare data structures, e.g., params_, layers_, etc.
   */
  void PrepareDataStructures();

 protected:
  std::vector<Layer*> layers_;
  std::vector<Param*> params_;

  std::map<std::string, Layer*> name2layer_;
  std::map<int, Param*> paramid2param_;
};

}  // namespace singa

#endif  // SINGA_NEURALNET_NEURALNET_H_
