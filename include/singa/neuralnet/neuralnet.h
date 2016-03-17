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
#include <unordered_map>

#include "singa/neuralnet/layer.h"
#include "singa/proto/job.pb.h"
#include "singa/utils/factory.h"
#include "singa/utils/graph.h"

namespace singa {
using std::unordered_map;
using std::string;
using std::vector;
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
   * Create for python binding, production test mode
   *
   */
  static NeuralNet* CreateNeuralNet(const string str);
  NeuralNet() {};
  
  static const NetProto Unrolling(const NetProto& net_conf);
  /**
   * construct the net structure from protocol buffer.
   * @param netproto neural net config
   * @param npartitions num of partitions. 1 for no partitioning.
   */
  NeuralNet(NetProto net_conf, int num_partitions);
  ~NeuralNet();
  /**
   * Load net params from checkpoint fiels.
   * @param path checkpoint files
   */
  void Load(const vector<string>& path);
  /**
   * load specified Param objects from from checkpoint files.
   *
   * Param objects and blobs are matched based on name.
   * The param from previous checkpoint files will be overwritten by
   * the param with the same name in later checkpoint files.
   *
   * @param[in] path
   * @param[in,out] params load Blobs with the same name as the Params in this
   * this dictionary. The Param values are copied into the corresponding Param
   * objects.
   */
  static void Load(const vector<string>& path,
                   const unordered_map<string, Param*>& params);
  /**
   * To display the adjacency layers
  std::string ToAdjacency();
   */
  /**
   * Share memory of parameter values from other neuralnet
   * @param[in] other the neural net from which to share the Params
   * @param[in] cpu_only if true only share cpu memory; else, share both cpu
   * and gpu memory.
   */
  void ShareParamsFrom(NeuralNet* other, bool cpu_only);
  inline const std::vector<Layer*>& layers() const { return layers_; }
  inline const std::vector<Param*>& params() const { return params_; }
  inline Layer* name2layer(std::string name) const {
    if (name2layer_.find(name) == name2layer_.end())
      return nullptr;
    else
      return name2layer_.at(name);
  }
  inline const std::vector<Layer*>& srclayers(const Layer* layer) const {
    CHECK(src_map_.find(layer) != src_map_.end())
      << "layer (" << layer->name() << " ) has no source layers";
    return src_map_.at(layer);
  }
  Layer* last_unroll_layer(const Layer* layer) const {
    auto pos = layer->name().find("#");
    if (pos == std::string::npos)
      return nullptr;
    string last_name = std::to_string(unroll_len_) + layer->name().substr(pos);
    CHECK(name2layer_.find(last_name) != name2layer_.end())
      << "layer name = " << last_name << " has no unroll layers";
    return name2layer_.at(last_name);
  }
  inline Param* paramid2param(int id) const { return paramid2param_.at(id); }

  /**
   * Conver the neural net into graph representation.
   * Each layer is converted into a node.
   * @param include_shape if true label the node with shape info
   */
  const Graph ToGraph(bool include_shape) const;

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
  Graph* CreateGraph(const NetProto& netproto, int num_partitions);
  /**
   * Create neural net from graph, one layer per node.
   */
  void CreateNetFromGraph(Graph* graph);
  /**
   * prepare data structures, e.g., params_, layers_, etc.
   */
  void PrepareDataStructures();
  void PrepareDataStructures(const NetProto& proto);
  /**
   * add split layers, due to connections to multiple dst-layers
   */
  NetProto AddModelSplitLayers(const NetProto& netproto);
  /**
   * add connection layers, due to partition of the whole nerualnet
   * this should be done after AddModelSplitLayers()
   */
  NetProto AddPartitionConnectionLayers(const NetProto& netproto,
                                        int npartitions);

 protected:
  int unroll_len_ = 1;
  std::vector<Layer*> layers_;
  std::vector<Param*> params_;

  unordered_map<std::string, Layer*> name2layer_;
  unordered_map<int, Param*> paramid2param_;
  unordered_map<const Layer*, std::vector<Layer*>> src_map_;
};

}  // namespace singa

#endif  // SINGA_NEURALNET_NEURALNET_H_
