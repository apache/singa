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

#include "singa/neuralnet/neuralnet.h"

#include <algorithm>
#include <queue>
#include "singa/utils/singleton.h"

namespace singa {

using std::map;
using std::string;
using std::vector;

NeuralNet* NeuralNet::Create(const NetProto& net_conf, Phase phase,
                                        int npartitions) {
  NetProto conf;
  conf.CopyFrom(net_conf);
  conf.clear_layer();
  // for sharing param conf
  std::unordered_map<string, ParamProto*> name2param;
  std::vector<ParamProto*> shares;
  // flag=0: neither exclude nor include field appears
  // flag=1: exclude field appears
  // flag=2: include field appears
  int flag = 0;
  // exclude layers according to phase
  // exclude field is deprecated
  // please use include field instead
  for (const auto& layer : net_conf.layer()) {
    bool include = true;
    for (auto p : layer.exclude()) {
      // check whether both exclude and include field
      // appear in the same .conf file
      CHECK(flag == 0 || flag == 1)
        << "include and exclude field should not simultaneously"
        << " appear in the same .conf file";
      if (p == phase)
        include = false;
      flag = 1;
    }
    // neural net only include the specified layer in the include field
    for (auto p : layer.include()) {
      // check whether both exclude and include field
      // appear in the same .conf file
      CHECK(flag == 0 || flag == 2)
        << "include and exclude field should not simultaneously"
        << " appear in the same .conf file";
      if (p == phase) {
        include = true;
        break;
      }
      include = false;
      flag = 2;
    }
    if (include == false) continue;
    LayerProto* layer_conf = conf.add_layer();
    layer_conf->CopyFrom(layer);
    // using net partition if layer partition is not set
    if (!layer_conf->has_partition_dim())
      layer_conf->set_partition_dim(net_conf.partition_dim());
    for (int i = 0; i < layer_conf->param_size(); i++) {
      ParamProto* param = layer_conf->mutable_param(i);
      if (param->has_name() && param->name() != "") {
        CHECK(name2param.find(param->name()) == name2param.end())
          << "param name is repeated: " << param->name();
        name2param[param->name()] = param;
      }
      if (param->has_share_from() && param->share_from() != "")
        shares.push_back(param);
    }
  }
  LOG(INFO) << "Before unrolling: \n" << conf.DebugString();
  conf = Unrolling (conf);
  LOG(INFO) << "After rolling: \n" << conf.DebugString();
  for (auto param : shares) {
    const std::string from = param->share_from();
    const std::string name = param->name();
    CHECK(name2param.find(from) != name2param.end())
      << "can't find param " << from;
    // CopyFrom will overwrite the name and share_from fields
    param->CopyFrom(*name2param.at(from));
    param->set_name(name);
    param->set_share_from(from);
  }
  LOG(INFO) << "NeuralNet config is\n" << conf.DebugString();
  // TODO(wangwei) create net based on net type, e.g., directed, undirected, etc
  return new NeuralNet(conf, npartitions);
}

NetProto NeuralNet::Unrolling(const NetProto& net_conf) {
	NetProto conf;
	conf.CopyFrom(net_conf);
	//conf.Clear();
	return conf;
}


NeuralNet::NeuralNet(NetProto netproto, int npartitions) {
  LOG(INFO) << "Constructing Neural Net...";
  auto graph = CreateGraph(netproto, npartitions);
  CreateNetFromGraph(graph, npartitions);
  PrepareDataStructures();
  for (Node* node : graph->nodes())
    delete static_cast<LayerProto*>(node->proto);
  delete graph;
  LOG(INFO) << "Neural net constructed";
}

NeuralNet::~NeuralNet() {
  for (auto layer : layers_)
    delete layer;
}
void NeuralNet::Load(const vector<string>& paths) {
  unordered_map<string, Param*> params;
  for (auto p : params_) {
    params[p->name()] = p;
  }
  Load(paths, params);
}
void NeuralNet::Load(const vector<string>& paths,
    const unordered_map<string, Param*>& params) {
  for (const auto path : paths) {
    LOG(ERROR) << "Load from checkpoint file " << path;
    BlobProtos bps;
    // TODO(wangwei) extend to read checkpoint from HDFS
    ReadProtoFromBinaryFile(path.c_str(), &bps);
    for (int i = 0; i < bps.name_size(); i++) {
      if (params.find(bps.name(i)) != params.end()) {
        params.at(bps.name(i))->FromProto(bps.blob(i));
        params.at(bps.name(i))->set_version(bps.version(i));
      }
    }
  }
}

/*
std::string NeuralNet::ToAdjacency() {
  string disp = "";
  for (auto& layer : layers_) {
    disp += layer->name()+": ";
    for (const auto& dst : layer->dstlayers())
      disp += dst->name()+", ";
    disp += "\n";
  }
  return disp;
}
*/

void NeuralNet::ShareParamsFrom(NeuralNet* other) {
  for (auto& layer : layers_) {
    auto otherlayer = other->name2layer(layer->name());
    if (otherlayer != nullptr) {
      const auto& otherparams = otherlayer->GetParams();
      const auto& params = layer->GetParams();
      CHECK_EQ(params.size(), otherparams.size());
      for (size_t i = 0; i < params.size(); i++) {
        params[i]->ShareFrom(*otherparams[i]);
      }
    }
  }
}

// add a node for SliceLayer between srcnode and dstnodes
Node* SliceNode(Graph* graph, Node* srcnode,
    const vector<Node*>& dstnodes, bool connect_dst) {
  string name = srcnode->name + "<";
  LayerProto *proto = new LayerProto();
  proto->set_name(name);
  proto->set_type(LayerType::kSlice);
  proto->set_partition_id(
      static_cast<LayerProto*>(srcnode->proto)->partition_id());
  auto conf = proto->mutable_slice_conf();
  conf->set_slice_dim(
      static_cast<LayerProto*>(dstnodes[0]->proto)->partition_dim());
  Node* node = new Node(name, "##" + name, proto->partition_id(), proto);
  graph->AddNode(node);
  graph->AddEdge(srcnode, node);
  if (connect_dst)
    for (Node* dst : dstnodes)
      graph->AddEdge(node, dst);
  return node;
}

// add a node for ConcateLayer between srcnodes and dstnode
Node* ConcateNodes(Graph* graph, const vector<Node*>& srcnodes, Node* dstnode) {
  string name = ">" + dstnode->name;
  LayerProto *proto = new LayerProto();
  proto->set_name(name);
  proto->set_type(LayerType::kConcate);
  proto->set_partition_id(
      static_cast<LayerProto*>(dstnode->proto)->partition_id());
  auto conf = proto->mutable_concate_conf();
  conf->set_concate_dim(
      static_cast<LayerProto*>(srcnodes[0]->proto)->partition_dim());
  Node* node = new Node(name, "##" + name, proto->partition_id(), proto);
  graph->AddNode(node);
  graph->AddEdge(node, dstnode);
  for (Node* src : srcnodes)
    graph->AddEdge(src, node);
  return node;
}

// add a node for SplitLayer between srcnode and dstnodes
Node* SplitNode(Graph* graph, Node* srcnode, const vector<Node*>& dstnodes) {
  string name = srcnode->name + "+";
  LayerProto *proto = new LayerProto();
  proto->set_name(name);
  proto->set_type(LayerType::kSplit);
  proto->set_partition_id(
      static_cast<LayerProto*>(srcnode->proto)->partition_id());
  Node* node = new Node(name, "##" + name, proto->partition_id(), proto);
  graph->AddNode(node);
  graph->AddEdge(srcnode, node);
  for (Node* dst : dstnodes)
    graph->AddEdge(node, dst);
  return node;
}

// add a pair of nodes for BridgeSrcLayer and BridgeDstLayer between srcnode
// and dstnode
void BridgeNodes(Graph* graph, Node* srcnode, Node* dstnode) {
  string sname = srcnode->name + ":-";
  LayerProto *sproto = new LayerProto();
  sproto->set_name(sname);
  sproto->set_type(LayerType::kBridgeSrc);
  sproto->set_partition_id(
      static_cast<LayerProto*>(srcnode->proto)->partition_id());
  auto sbridge = new Node(sname, "##" + sname, sproto->partition_id(), sproto);
  string dname = "-:" + dstnode->name;
  LayerProto *dproto = new LayerProto();
  dproto->set_name(dname);
  dproto->set_type(LayerType::kBridgeDst);
  dproto->set_partition_id(
      static_cast<LayerProto*>(dstnode->proto)->partition_id());
  auto dbridge = new Node(dname, "##" + dname, dproto->partition_id(), dproto);
  graph->AddNode(sbridge);
  graph->AddNode(dbridge);
  graph->AddEdge(srcnode, sbridge);
  graph->AddEdge(sbridge, dbridge);
  graph->AddEdge(dbridge, dstnode);
}

Graph* NeuralNet::CreateGraph(const NetProto& netproto, int npartitions) {
  Graph *graph = new Graph();
  // from name of original layer to nodes
  map<string, vector<Node*>> name2nodes;
  map<string, const LayerProto*> name2proto;
  for (const auto& layer : netproto.layer()) {
    vector<Node*> nodes;
    int pdim = layer.partition_dim();
    if (pdim == 0 || pdim == 1) {
      char suffix[4];
      for (int i = 0; i < npartitions; i++) {
        LayerProto *proto = new LayerProto(layer);
        snprintf(suffix, sizeof(suffix), "%02d", i);
        // differentiate partitions
        string nodename = layer.name() + "@" + string(suffix);
        proto->set_partition_id(i);
        proto->set_num_partitions(npartitions);
        proto->set_name(nodename);
        auto node = new Node(nodename, layer.name(), i, proto);
        graph->AddNode(node);
        nodes.push_back(node);
      }
    } else if (pdim == -1) {
      LayerProto *proto = new LayerProto(layer);
      auto node = new Node(layer.name(), layer.name(), 0, proto);
      graph->AddNode(node);
      nodes.push_back(node);
    } else {
      LOG(FATAL) << "Cannot partition layer (" << layer.name() <<") on dim: "
        << layer.partition_dim();
    }
    name2nodes[layer.name()] = nodes;
    name2proto[layer.name()] = &layer;
  }

  // connect nodes, nodes for ConcateLayer, SliceLayer and SplitLayer are added.
  for (const auto& layerproto : netproto.layer()) {
    string name = layerproto.name();
    int pdim = layerproto.partition_dim();
    const vector<Node*>& nodes = name2nodes.at(name);
    for (auto srcname : layerproto.srclayers()) {
      const vector<Node*>& srcnodes = name2nodes.at(srcname);
      // TODO(wangwei): consider the type of each connection
      Layer *layer = Layer::Create(layerproto);
      ConnectionType connection = layer->src_neuron_connection(0);
      delete layer;
      int src_pdim = name2proto[srcname]->partition_dim();
      // no partition of src layer
      if (src_pdim == -1) {
        Node* srcnode = srcnodes[0];
        if (pdim == 0 || (pdim == 1 && connection == kOneToOne))
          SliceNode(graph, srcnode, nodes, true);
        else if (pdim == -1)
          graph->AddEdge(srcnode, nodes[0]);
        else  // type==kLayerPartition&&connection==kOneToAll
          SplitNode(graph, srcnode, nodes);
      } else if ((pdim == -1 && (src_pdim == 0 || src_pdim == 1))
          ||(pdim == 1 && connection == kOneToAll && src_pdim == 0)) {
        // copy/concate the whole srclayer for every dst partition
        for (Node* node : nodes)
          ConcateNodes(graph, srcnodes, node);
      } else if ((src_pdim == 1 && pdim == 0) || (src_pdim == 0 && pdim == 1)) {
        // TODO(wangwei) rewrite the whole src-dst construction in a clear way
        LOG(FATAL) << "not implemented";
        // the most complext scenario
        // vector<Node*> nodes;
        // for (Node* srcnode : srcnodes)
        //   nodes.push_back(SliceNode(graph, srcnode, nodes, false));
        // for (Node* node : nodes)
        //   ConcateNodes(graph, nodes, node);
      } else if ((src_pdim == 0 && pdim == 0)||
          (src_pdim == 1 && pdim == 1 && connection == kOneToOne)) {
        CHECK_EQ(srcnodes.size(), nodes.size());
        for (size_t i = 0; i < srcnodes.size(); i++)
          graph->AddEdge(srcnodes[i], nodes[i]);
      } else {
        LOG(FATAL) << "in wrong branch, not implemented";
      }
    }
  }
  // must do topology sort, because we have added new nodes.
  graph->Sort();
  // add nodes for SplitLayer
  vector<Node*> oldnodes = graph->nodes();
  for (Node* node : oldnodes) {
    auto layer = Layer::Create(*static_cast<LayerProto*>(node->proto));
    if (node->dstnodes.size() > 1
        && layer->dst_layer_connection() == kOneToOne) {
      vector<Node*> dstnodes = node->dstnodes;
      for (Node* dst : dstnodes)
        graph->RemoveEdge(node, dst);
      SplitNode(graph, node, dstnodes);
    }
    delete layer;
  }
  // add nodes for bridge layers
  for (Node* node : oldnodes) {
    vector<Node*> dstnodes = node->dstnodes;
    auto pid1 = static_cast<LayerProto*>(node->proto)->partition_id();
    for (size_t i = 0; i < dstnodes.size(); i++) {
      Node* dstnode = dstnodes.at(i);
      auto pid2 = static_cast<LayerProto*>(node->proto)->partition_id();
      if (pid1 != pid2) {
        graph->RemoveEdge(node, dstnode);
        BridgeNodes(graph, node, dstnode);
      }
    }
  }
  graph->Sort();
  DLOG(INFO) << "Pure graph structure\n" << graph->ToJson();
  return graph;
}

void NeuralNet::CreateNetFromGraph(Graph* graph, int npartitions) {
  // create one layer per node
  for (Node* node : graph->nodes()) {
    auto proto_ptr = static_cast<LayerProto*>(node->proto);
    auto layer = Layer::Create(*proto_ptr);
    layers_.push_back(layer);
    name2layer_[node->name] = layer;
  }
  // connect layers
  for (Node* node : graph->nodes()) {
    auto layer = name2layer(node->name);
    src_map_[layer] = vector<Layer*>{};
    for (Node* src : node->srcnodes)
      src_map_[layer].push_back(name2layer(src->name));
  }

  // setup layers
  int paramid = 0;
  map<string, string> layerinfo;
  map<string, vector<Layer*>> share_param_layers;
  for (Node* node : graph->nodes()) {
    auto layer = name2layer(node->name);
    layer->Setup(*(static_cast<LayerProto*>(node->proto)), srclayers(layer));
    LOG(INFO) << "constructing graph: " << layer->name();
    layerinfo[layer->name()] = IntVecToString(layer->data(nullptr).shape());
    string param_name = "$";
    for (auto param : layer->GetParams()) {
      param->set_id(paramid++);
      // if user does not name the param, then name it based on layer name.
      if (param->name() == "") {
        param->set_name(layer->name() + param_name);
        param_name += "$";
      }
    }
    if (layer->partition_dim() == 0)
      share_param_layers[node->origin].push_back(layer);
  }
  LOG(INFO) << "Neural net structure\n"  << graph->ToJson(layerinfo);
  // create map from param name to param ptr
  std::unordered_map<string, Param*> name2param;
  for (auto layer : layers_) {
    for (auto param : layer->GetParams()) {
      name2param[param->name()] = param;
    }
  }
  for (auto & entry : share_param_layers) {
    // overwrite entries for replicated params due to layer partition (dim 0).
    for (auto *param : entry.second.front()->GetParams())
      name2param.at(param->name()) = param;
  }
  // share params based on share_from field
  for (auto & entry : name2param) {
    Param* param = entry.second;
    const string share_from = param->share_from();
    if (param->share_from() != "") {
      if (name2param.find(share_from) != name2param.end()) {
        param->ShareFrom(*name2param.at(param->share_from()));
      } else {
        LOG(FATAL) << "No param with the name (share_from) " << share_from;
      }
    }
  }
  // share Params for layers generated (partitioned) from the same origin layer
  for (auto & entry : share_param_layers) {
    const auto& owner = entry.second.begin();
    const auto& owner_params = (*owner)->GetParams();
    for (auto it = owner + 1; it != entry.second.end(); it++) {
      auto params = (*it)->GetParams();
      CHECK_EQ(params.size(), owner_params.size());
      for (size_t i = 0; i < params.size(); i++)
        params.at(i)->ShareFrom(*owner_params.at(i));
    }
  }
}

void NeuralNet::PrepareDataStructures() {
  params_.clear();
  paramid2param_.clear();
  name2layer_.clear();
  for (auto& layer : layers_) {
    name2layer_[layer->name()] = layer;
    for (Param* p : layer->GetParams()) {
      paramid2param_[p->id()] = p;
      params_.push_back(p);
    }
  }
}

}  // namespace singa
