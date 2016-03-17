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
#include <unordered_map>
#include <algorithm>
#include <queue>
#include "singa/utils/singleton.h"

namespace singa {

using std::map;
using std::string;
using std::vector;

/**
 * Check user defined net config and make some preprocessing, e.g., assing names
 * to params.
 * TODO(wnagwei) implement the following functions.
 * 1. layer and paramname should not include '@', '+' and '#'. '@<suffix>'
 * is used for identifying layer location/partition. '<prefix>#' is used for
 * identifying the unrolled Param in RNN models.
 * 2. assign names to unnamed Param, e.g., p<param_id>+<layer_name>.
 */
const NetProto NetConfPreprocess(const NetProto& conf) {
  /*
  string param_name = "$";
  // if user does not name the param, then name it based on layer name.
  if (param->name() == "") {
    param->set_name(layer->name() + param_name);
    param_name += "$";
  }
  */
  NetProto proto = conf;
  for (int i = 0; i < proto.layer_size(); i++) {
    if (!proto.layer(i).has_unroll_len())
      proto.mutable_layer(i)->set_unroll_len(proto.unroll_len());
  }
  return proto;
}

NeuralNet* NeuralNet::CreateNeuralNet(const string str) {
  NetProto net_conf;
  net_conf.ParseFromString(str);
  return NeuralNet::Create(net_conf,singa::kTest,1);
}

NeuralNet* NeuralNet::Create(const NetProto& net_conf, Phase phase,
    int npartitions) {
  const NetProto& full_net_conf = NetConfPreprocess(net_conf);
  NetProto conf = full_net_conf;
  conf.clear_layer();
  // flag=0: neither exclude nor include field appears
  // flag=1: exclude field appears
  // flag=2: include field appears
  int flag = 0;
  // exclude layers according to phase
  // exclude field is deprecated
  // please use include field instead
  for (const auto& layer : full_net_conf.layer()) {
    bool include = true;
    for (auto p : layer.exclude()) {
      // check whether both exclude and include field
      // appear in the same .conf file
      CHECK(flag == 0 || flag == 1) << "Don't use include and exclude together";
      if (p == phase)
        include = false;
      flag = 1;
    }
    // neural net only include the specified layer in the include field
    for (auto p : layer.include()) {
      CHECK(flag == 0 || flag == 2) << "Don't use include and exclude together";
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
  }
  // LOG(INFO) << "Before unrolling: \n" << conf.DebugString();
  conf = Unrolling(conf);

  // Copy shared parameters for sharing param conf
  std::vector<ParamProto*> shares;
  std::unordered_map<string, ParamProto*> name2param;
  for (int index = 0; index < conf.layer_size(); index++) {
    LayerProto* layer = conf.mutable_layer(index);
    for (int i = 0; i < layer->param_size(); i++) {
      ParamProto* param = layer->mutable_param(i);
      CHECK(name2param.find(param->name()) == name2param.end())
        << "Repeated param = " << param->name();
      name2param[param->name()] = param;
      if (param->has_share_from() && param->share_from() != "")
        shares.push_back(param);
    }
  }
  for (auto param : shares) {
    const std::string from = param->share_from();
    const std::string name = param->name();
    CHECK(name2param.find(from) != name2param.end())
      << "can't find share_from = " << from;
    // CopyFrom will overwrite the name and share_from fields
    param->CopyFrom(*name2param.at(from));
    param->set_name(name);
    param->set_share_from(from);
  }
  LOG(INFO) << "Initial NeuralNet Config is\n" << conf.DebugString();
  // TODO(wangwei) create net based on net type, e.g., directed, undirected.
  return new NeuralNet(conf, npartitions);
}

const NetProto NeuralNet::Unrolling(const NetProto& net_conf) {
  // Step 1: Unroll each layer & set parameter sharing
  NetProto conf;

  std::vector<std::vector<int>> layer_groups;
  std::unordered_map<string, int> org_layer_names;
  for (int index = 0; index < net_conf.layer_size(); index ++) {
    const LayerProto& org_layer = net_conf.layer(index);
    org_layer_names[org_layer.name()] = index;  // layer_name -> index

    std::vector<int> layer_group;
    for (int i = 0; i < org_layer.unroll_len(); i ++) {  // unroll
      LayerProto* unroll_layer = conf.add_layer();
      unroll_layer->CopyFrom(org_layer);  // create a new layer conf
      // update layer names
      std::stringstream sstm;
      sstm << i << '#' << unroll_layer->name();
      unroll_layer->set_name(sstm.str());
      unroll_layer->set_unroll_index(i);
      // update layer parameter sharing
      for (int j = 0; j < unroll_layer->param_size(); j ++) {
        ParamProto* param = unroll_layer->mutable_param(j);
        if (i > 0) {
          param->set_share_from("0#" + param->name());
        }
        std::stringstream sstm1;
        sstm1 << i << '#' << param->name();
        param->set_name(sstm1.str());
      }
      // clear unrolling related fields
      unroll_layer->clear_unroll_len();
      unroll_layer->clear_unroll_conn_type();
      unroll_layer->clear_shift();
      unroll_layer->clear_srclayers();

      layer_group.push_back(conf.layer_size() - 1);
      // LOG(ERROR) << "unrolling layer " << unroll_layer->name();
    }
    layer_groups.push_back(layer_group);
  }
  // Step 2: Connect unrolled layers by setting `srclayers`
  for (int index = 0; index < net_conf.layer_size(); index ++) {
    const LayerProto& org_layer = net_conf.layer(index);
    if (org_layer.srclayers_size() == 0)
      continue;   // no src layer
    for (int i = 0; i < org_layer.srclayers_size(); i ++) {
      const string& org_layer_src = org_layer.srclayers(i);
      singa::UnrollConnType unroll_conn_type = kUnrollOneToOne;
      if (i < org_layer.unroll_conn_type_size())
        unroll_conn_type = org_layer.unroll_conn_type(i);
      unsigned int shift = 0;
      if (i < org_layer.shift_size())
        shift = org_layer.shift(i);

      const std::vector<int> unroll_layer_srcs
        = layer_groups[org_layer_names[org_layer_src]];

      for (unsigned int j = 0; j < layer_groups[index].size(); j ++) {
        LayerProto* unroll_layer = conf.mutable_layer(layer_groups[index][j]);
        // Update src layers of `unroll_layer` by considering the types
        if (unroll_conn_type == kUnrollOneToAll) {
          for (int unroll_layer_src : unroll_layer_srcs) {
            unroll_layer->add_srclayers(conf.layer(unroll_layer_src).name());
          }
        } else if (unroll_conn_type == kUnrollOneToOne) {
          if (j < shift) continue;  // no need to connect with the src
          int unroll_layer_src = unroll_layer_srcs[j - shift];
          unroll_layer->add_srclayers(conf.layer(unroll_layer_src).name());
        } else if (unroll_conn_type == kUnrollFirstToLast) {
          if (j > 0) break;
          int unroll_layer_src =
            unroll_layer_srcs[unroll_layer_srcs.size() - 1];
          unroll_layer->add_srclayers(conf.layer(unroll_layer_src).name());
        }
      }
    }

    // TODO(fanju): add LSTM when it is ready
    if (org_layer.type() == kGRU) {  // connect GRU layers
      for (unsigned int j = 1; j < layer_groups[index].size(); j ++) {
        LayerProto* unroll_layer = conf.mutable_layer(layer_groups[index][j]);
        string srcname = conf.layer(layer_groups[index][j-1]).name();
        unroll_layer->add_srclayers(srcname);
      }
    }
  }
  return conf;
}


NeuralNet::NeuralNet(NetProto netproto, int npartitions) {
  LOG(INFO) << "Constructing NeuralNet...";
  auto graph = CreateGraph(netproto, npartitions);
  CreateNetFromGraph(graph);
  PrepareDataStructures();

  for (Node* node : graph->nodes())
    delete static_cast<LayerProto*>(node->proto);
  delete graph;
  LOG(INFO) << "NeuralNet Constructed";
  unroll_len_ = netproto.unroll_len();
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
        // LOG(ERROR) << "Loading param = " << bps.name(i);
        params.at(bps.name(i))->FromProto(bps.blob(i));
        params.at(bps.name(i))->set_version(bps.version(i));
      }
    }
  }
}

void NeuralNet::ShareParamsFrom(NeuralNet* other, bool cpu_only) {
  for (auto& layer : layers_) {
    auto otherlayer = other->name2layer(layer->name());
    if (otherlayer != nullptr) {
      const auto& otherparams = otherlayer->GetParams();
      const auto& params = layer->GetParams();
      CHECK_EQ(params.size(), otherparams.size());
      for (size_t i = 0; i < params.size(); i++) {
        params[i]->ShareDataFrom(otherparams[i], cpu_only);
      }
    }
  }
}

// name of connection layers
string splitName(const string& layer) { return "split("+layer+")"; }
string sliceName(const string& layer) { return "slice("+layer+")"; }
string concateName(const string& layer) { return "concate("+layer+")"; }
string bridgeName(const string& src, const string& dst) { return src+"->"+dst; }
string bridgeSrcName(const string& src, const string& dst) {
  return "bridge_src("+bridgeName(src, dst)+")";
}
string bridgeDstName(const string& src, const string& dst) {
  return "bridge_dst("+bridgeName(src, dst)+")";
}

ConnectionType dstLayerConnection(const LayerProto& proto) {
  auto layer = Layer::Create(proto);
  auto ret = layer->dst_layer_connection();
  delete layer;
  return ret;
}

ConnectionType srcNeuronConnection(const LayerProto& proto) {
  auto layer = Layer::Create(proto);
  auto ret = layer->src_neuron_connection(0);
  delete layer;
  return ret;
}

NetProto NeuralNet::AddModelSplitLayers(const NetProto& netproto) {
  NetProto net_w_split;
  net_w_split.CopyFrom(netproto);
  net_w_split.clear_layer();
  // calculate number of dst-layers for each layer
  map<string, int> dst_count;
  for (const LayerProto& layer : netproto.layer())
    for (const string& src_name : layer.srclayers())
      ++dst_count[src_name];
  // tag to add split layer if:
  // dst_count[] > 1 && dst_layer_connection() = OneToOne
  for (const LayerProto& layer : netproto.layer())
    if ((dst_count[layer.name()] > 1 && dstLayerConnection(layer) == kOneToOne))
        dst_count[layer.name()] = -dst_count[layer.name()];
  // add orginal layers and adjust srclayers
  for (const LayerProto& layer : netproto.layer()) {
    LayerProto* proto = net_w_split.add_layer();
    proto->CopyFrom(layer);
    proto->clear_srclayers();
    for (const string& src_name : layer.srclayers())
      if (dst_count[src_name] < 0)
        proto->add_srclayers(splitName(src_name));
      else
        proto->add_srclayers(src_name);
  }
  // add split layers
  for (const LayerProto& layer : netproto.layer()) {
    if (dst_count[layer.name()] < 0) {
      LayerProto* split_proto = net_w_split.add_layer();
      split_proto->set_name(splitName(layer.name()));
      split_proto->set_type(kSplit);
      split_proto->set_partition_dim(layer.partition_dim());
      split_proto->add_srclayers(layer.name());
      split_proto->mutable_split_conf()
                 ->set_num_splits(-dst_count[layer.name()]);
    }
  }
  // LOG(INFO) << "NeuralNet Config After Model Split is\n"
  //           << net_w_split.DebugString();
  return net_w_split;
}

NetProto NeuralNet::AddPartitionConnectionLayers(const NetProto& netproto,
                                                 int npartitions) {
  CHECK_GT(npartitions, 0);
  NetProto net_w_connection;
  net_w_connection.CopyFrom(netproto);
  // if npartitions is 1, no need to add connection layers
  if (npartitions == 1) return net_w_connection;
  // add original layers, but remove all edges first
  net_w_connection.clear_layer();
  map<string, LayerProto*> name2proto;
  for (const LayerProto& layer : netproto.layer()) {
    LayerProto* layer_proto = net_w_connection.add_layer();
    layer_proto->CopyFrom(layer);
    layer_proto->clear_srclayers();
    name2proto[layer_proto->name()] = layer_proto;
  }
  /*
   * Add Slice, Concate, Split Layers for Model Partition
   *
   * All cases are as follows:
   * src_pdim | dst_pdim | connection_type | Action
   *     0    |     0    |     OneToOne    | Direct Connection
   *     1    |     1    |     OneToOne    | Direct Connection
   *     0    |     0    |     OneToAll    | Direct Connection
   *     1    |     0    |     OneToOne    | Slice -> Concate
   *     0    |     1    |     OneToOne    | Slice -> Concate
   *     1    |     0    |     OneToAll    | Slice -> Concate
   *     0    |     1    |     OneToAll    | Split -> Concate
   *     1    |     1    |     OneToAll    | Split -> Concate
   *
   * Logic:
   * dst_pdim = 1 && OneToAll ?
   *   (YES) Split -> Concate
   *   (NO)  src_pdim = dst_pdim ?
   *           (YES) Direct Connection
   *           (NO)  Slice -> Concate
   */
  for (const LayerProto& origin_layer : netproto.layer()) {
    LayerProto* dst_layer = name2proto[origin_layer.name()];
    int dst_pdim = dst_layer->partition_dim();
    ConnectionType connection = srcNeuronConnection(*dst_layer);
    for (const string& src_name : origin_layer.srclayers()) {
      LayerProto* src_layer = name2proto[src_name];
      int src_pdim = src_layer->partition_dim();
      // dst_pdim = 1 && OneToAll ?
      if (dst_pdim == 1 && connection == kOneToAll) {
        // add split layer
        LayerProto* split_layer = net_w_connection.add_layer();
        split_layer->set_name(splitName(src_layer->name()));
        split_layer->set_type(kSplit);
        split_layer->set_partition_dim(src_layer->partition_dim());
        split_layer->add_srclayers(src_layer->name());
        split_layer->mutable_split_conf()->set_num_splits(npartitions);
       // add concate layer
       LayerProto* concate_layer = net_w_connection.add_layer();
       concate_layer->set_name(concateName(split_layer->name()));
       concate_layer->set_type(kConcate);
       concate_layer->set_partition_dim(dst_layer->partition_dim());
       // concate on src_pdim
       concate_layer->mutable_concate_conf()
         ->set_concate_dim(src_layer->partition_dim());
       concate_layer->mutable_concate_conf()->set_num_concates(npartitions);
       concate_layer->add_srclayers(split_layer->name());
       // connect dst_layer to concate layer
       dst_layer->add_srclayers(concate_layer->name());
      } else {
        // src_pdim = dst_pdim ?
        if (dst_pdim == src_pdim) {
          // direct connection
          dst_layer->add_srclayers(src_layer->name());
        } else {
          // add slice layer
          LayerProto* slice_layer = net_w_connection.add_layer();
          slice_layer->set_name(sliceName(src_layer->name()));
          slice_layer->set_type(kSlice);
          slice_layer->set_partition_dim(src_layer->partition_dim());
          // slice on dst_pdim
          slice_layer->mutable_slice_conf()
            ->set_slice_dim(dst_layer->partition_dim());
          slice_layer->mutable_slice_conf()->set_num_slices(npartitions);
          slice_layer->add_srclayers(src_layer->name());
          // add concate layer
          LayerProto* concate_layer = net_w_connection.add_layer();
          concate_layer->set_name(concateName(slice_layer->name()));
          concate_layer->set_type(kConcate);
          concate_layer->set_partition_dim(dst_layer->partition_dim());
          // concate on src_pdim
          concate_layer->mutable_concate_conf()
            ->set_concate_dim(src_layer->partition_dim());
          concate_layer->mutable_concate_conf()->set_num_concates(npartitions);
          concate_layer->add_srclayers(slice_layer->name());
          // connect dst_layer to concate layer
          dst_layer->add_srclayers(concate_layer->name());
        }
      }
    }
  }
  LOG(INFO) << "NeuralNet Config After Adding Connection Layers is\n"
            << net_w_connection.DebugString();
  return net_w_connection;
}

Graph* NeuralNet::CreateGraph(const NetProto& netproto, int npartitions) {
  NetProto net_w_split = AddModelSplitLayers(netproto);
  NetProto net_w_connection =
    AddPartitionConnectionLayers(net_w_split, npartitions);
  // for each original layer proto, create #npartitions of nodes
  Graph* graph = new Graph();
  map<string, vector<Node*>> name2nodes;
  map<string, const LayerProto*> name2proto;
  for (const LayerProto& layer : net_w_connection.layer()) {
    vector<Node*> nodes;
    for (int i = 0; i < npartitions; i++) {
      LayerProto *proto = new LayerProto(layer);
      // differentiate partitions
      string nodename = layer.name() + "@" + std::to_string(i);
      proto->set_name(nodename);
      proto->set_type(layer.type());
      proto->set_partition_dim(layer.partition_dim());
      proto->set_partition_id(i);
      proto->set_num_partitions(npartitions);
      Node* node = graph->AddNode(nodename, layer.name(), i, proto);
      nodes.push_back(node);
      // TODO(wangwei) update param name
    }
    name2nodes[layer.name()] = nodes;
    name2proto[layer.name()] = &layer;
  }
  // connect layers, add bridge layers if partition id is different
  for (const LayerProto& origin_layer : net_w_connection.layer()) {
    vector<Node*> dst_nodes = name2nodes[origin_layer.name()];
    for (const string& src_name : origin_layer.srclayers()) {
      vector<Node*> src_nodes = name2nodes[src_name];
      if (origin_layer.type() != kConcate) {
        for (size_t i = 0; i < src_nodes.size(); ++i) {
          CHECK_EQ(src_nodes[i]->partition_id, i);
          CHECK_EQ(dst_nodes[i]->partition_id, i);
          graph->AddEdge(src_nodes[i], dst_nodes[i]);
        }
      } else {
        // need to add bridge layers
        for (size_t i = 0; i < src_nodes.size(); ++i) {
          CHECK_EQ(src_nodes[i]->partition_id, i);
          for (size_t j = 0; j < dst_nodes.size(); ++j) {
            CHECK_EQ(dst_nodes[j]->partition_id, j);
            if (i == j) {  // in same partition, no bridge needed
              graph->AddEdge(src_nodes[i], dst_nodes[j]);
            } else {  // add bridges
              // bridge src && dst layer
              LayerProto *proto_bsrc = new LayerProto();
              LayerProto *proto_bdst = new LayerProto();
              string bsrc_name = bridgeSrcName(src_nodes[i]->name,
                                               dst_nodes[j]->name);
              string bdst_name = bridgeDstName(src_nodes[i]->name,
                                               dst_nodes[j]->name);
              proto_bsrc->set_name(bsrc_name);
              proto_bdst->set_name(bdst_name);
              proto_bsrc->set_type(kBridgeSrc);
              proto_bdst->set_type(kBridgeDst);
              proto_bsrc->set_partition_dim(origin_layer.partition_dim());
              proto_bdst->set_partition_dim(origin_layer.partition_dim());
              proto_bsrc->set_partition_id(src_nodes[i]->partition_id);
              proto_bdst->set_partition_id(dst_nodes[j]->partition_id);
              proto_bsrc->set_num_partitions(npartitions);
              proto_bdst->set_num_partitions(npartitions);
              Node* bsrc_node = graph->AddNode(bsrc_name, bsrc_name, i,
                                               proto_bsrc);
              Node* bdst_node = graph->AddNode(bdst_name, bdst_name, j,
                                               proto_bdst);
              graph->AddEdge(src_nodes[i], bsrc_node);
              graph->AddEdge(bsrc_node, bdst_node);
              graph->AddEdge(bdst_node, dst_nodes[j]);
            }
          }
        }
      }
    }
  }
  graph->Sort();
  // DLOG(INFO) << "Pure graph structure\n" << graph->ToJson();
  return graph;
}

void NeuralNet::CreateNetFromGraph(Graph* graph) {
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
    LOG(INFO) << "constructing graph: " << node->name;
    auto layer = name2layer(node->name);
    layer->Setup(*(static_cast<LayerProto*>(node->proto)), srclayers(layer));
    DLOG(INFO) << "constructing graph: " << layer->name();
    layerinfo[layer->name()] = IntVecToString(layer->data(nullptr).shape());
    for (auto param : layer->GetParams()) {
      param->set_id(paramid++);
    }
    if (layer->partition_dim() == 0)
      share_param_layers[node->origin].push_back(layer);
  }
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
        param->ShareDataFrom(name2param.at(param->share_from()), false);
      } else {
        LOG(FATAL) << "No param with the name (share_from) " << share_from;
      }
    }
  }

  // share params due to laye unrolling
  for (auto & entry : name2param) {
    Param* param = entry.second;
    auto pos = param->name().find("#");
    if (pos != std::string::npos && param->owner() != param->id()) {
      string from = "0" + param->name().substr(pos);
      CHECK(name2param.find(from) != name2param.end())
        << "Can't find owner = " << from << " for param = " << param->name();
      Param* owner = name2param.at(from);
      param->ShareFrom(owner);
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
        params.at(i)->ShareDataFrom(owner_params.at(i), true);
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

const Graph NeuralNet::ToGraph(bool include_shape) const {
  Graph g;
  map<string, string> attrs;
  attrs["shape"] = "box";
  vector<string> colors {"black", "red", "yellow", "blue"};
  for (auto layer : layers_) {
    LOG_IF(WARNING, layer->partition_id() >= static_cast<int>(colors.size()))
      << "Too many partitions for displaying";
    attrs["color"] = colors[layer->partition_id() % colors.size()];
    if (include_shape) {
      attrs["label"] = "shape: ";
      for (const auto& x : layer->data(nullptr).shape())
        attrs["label"] += std::to_string(x) + " ";
    }
    g.AddNode(layer->name(), attrs);
  }

  for (auto layer : layers_)
    for (auto src : src_map_.at(layer))
      g.AddEdge(src->name(), layer->name());
  return g;
}
}  // namespace singa
