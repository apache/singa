#include <algorithm>
#include <queue>

#include "proto/model.pb.h"
#include "neuralnet/neuralnet.h"
#include "utils/singleton.h"
#include "utils/factory.h"
#include "utils/graph.h"
#include "utils/cluster.h"

namespace singa {
#define LayerT(x) LayerProto_LayerType_k##x

#define RegisterLayer(factory, id) \
  factory->Register(LayerProto_LayerType_k##id,\
      CreateInstance(id##Layer, Layer))

void NeuralNet::RegisterLayers(){
  Factory<Layer>* factory=Singleton<Factory<Layer>>::Instance();
  RegisterLayer(factory, BridgeDst);
  RegisterLayer(factory, BridgeSrc);
  RegisterLayer(factory, Convolution);
  RegisterLayer(factory, Concate);
  RegisterLayer(factory, Dropout);
  RegisterLayer(factory, InnerProduct);
  RegisterLayer(factory, Label);
  RegisterLayer(factory, LMDBData);
  RegisterLayer(factory, LRN);
  RegisterLayer(factory, Mnist);
  RegisterLayer(factory, Prefetch);
  RegisterLayer(factory, Pooling);
  RegisterLayer(factory, RGBImage);
  RegisterLayer(factory, ReLU);
  RegisterLayer(factory, ShardData);
  RegisterLayer(factory, Slice);
  RegisterLayer(factory, SoftmaxLoss);
  RegisterLayer(factory, Split);
  RegisterLayer(factory, Tanh);
  RegisterLayer(factory, DBMBottom);
  RegisterLayer(factory, DBMTop);
}
shared_ptr<NeuralNet> NeuralNet::SetupNeuralNet(const NetProto& np, Phase phase,
    int group_size){
  NetProto proto;
  proto.set_partition_type(np.partition_type());
  // exclude layers if necessary
  for(auto& layer:np.layer()){
    bool include=true;
    for(int x: layer.exclude()){
      if(x==phase)
        include=false;
    }
    if(include){
      LayerProto* lp=proto.add_layer();
      lp->CopyFrom(layer);
    }
  }
  LOG(INFO)<<"NeuralNet config is "<<proto.DebugString();
  return make_shared<NeuralNet>(proto, group_size);
}
NeuralNet::NeuralNet(NetProto net_proto, int group_size) {
  group_size_=group_size;
  for(int i=0;i<net_proto.layer_size();i++){
    LayerProto * layer_proto=net_proto.mutable_layer(i);
    if(!layer_proto->has_partition_type())
      layer_proto->set_partition_type(net_proto.partition_type());
  }

  LOG(INFO)<<"Construct Neural Net...";
  ConstructNeuralNet(net_proto);
  {
    string vis_folder=Cluster::Get()->vis_folder();
    std::ofstream fout(vis_folder+"/nopartition.json", std::ofstream::out);
    fout<<ToString();
    fout.flush();
    fout.close();
  }
  if(group_size_>1){
    PartitionNeuralNet();
    string vis_folder=Cluster::Get()->vis_folder();
    std::ofstream fout(vis_folder+"/partition.json", std::ofstream::out);
    fout<<ToString();
    fout.flush();
    fout.close();
  }
  for(auto layer: layers_){
    DLOG(INFO)<<layer->name();
  }
  for(auto& layer: layers_){
    for(shared_ptr<Param> p: layer->GetParams()){
      params_.push_back(p);
    }
  }
  LOG(INFO)<<"Neural Net constructed";
  // init all data members to avoid conflicts from multi-thread access
  losslayers();
  paramid2param(0);
  datalayers();
  parserlayers();
}

void NeuralNet::ConstructNeuralNet(const NetProto& net_proto){
  // construct graph, one node for one layer, identified by layer name
  map<string, LayerProto> protos;
  for (auto &layer_proto : net_proto.layer()){
    graph_.AddNode(layer_proto.name());
    protos[layer_proto.name()]=layer_proto;
  }
  for (auto &layer_proto : net_proto.layer())
    if(layer_proto.srclayers_size())
      for(const string& src: layer_proto.srclayers())
        graph_.AddEdge(src, layer_proto.name());

  // topology sort
  graph_.Sort();
  //LOG(ERROR)<<"pure graph without partition\n"<< graph_.ToString();

  auto* factory=Singleton<Factory<Layer>>::Instance();
  // create Layers according to topology order
  for(SNode node: graph_.nodes()){
    shared_ptr<Layer> layer(factory->Create(protos[node->name()].type()));
    layer->Init(protos[node->name()]);
    name2layer_[node->name()]=layer;
    layers_.push_back(layer);
  }

  // connect Layers.
  for(SNode node: graph_.nodes()){
    auto layer=name2layer_[node->name()];
    for(SNode dst: node->dstnodes())
      layer->AddDstLayer(name2layer_[dst->name()]);
    for(SNode src: node->srcnodes())
      layer->AddSrcLayer(name2layer_[src->name()]);
  }
  // setup layer properties, e.g., shapes
  int paramid=0;
  for(auto& layer: layers_){
      layer->Setup();
      for(auto param: layer->GetParams())
        param->set_id(paramid++);
  }
  LOG(INFO)<<"network graph witout partition\n"<<ToString();
}

void NeuralNet::PartitionNeuralNet(){
  graph_=CreatePartitonedGraph(layers_, name2layer_);
  //DLOG(ERROR)<<"pure graph after partition\n"<<graph_.ToString();
  map<string, shared_ptr<Layer>> name2layer(name2layer_);
  map<string, vector<shared_ptr<Layer>>> share_conf_layers;
  name2layer_.clear();
  layers_.clear();
  int gsize=group_size_;
  auto* factory=Singleton<Factory<Layer>>::Instance();
  // create Layers according to topology order
  for(SNode node: graph_.nodes()){
    LayerProto proto;
    proto.set_name(node->name());
    proto.set_partitionid(node->val().partitionid);
    string origin=node->val().origin;
    if (origin=="kSlice"){
      proto.set_type(LayerT(Slice));
      SliceProto *slice=proto.mutable_slice_conf();
      slice->set_slice_dimension(node->val().slice_dimension);
      slice->set_slice_num(node->dstnodes().size());
    }else if(origin== "kConcate"){
      proto.set_type(LayerT(Concate));
      ConcateProto *concate=proto.mutable_concate_conf();
      concate->set_concate_dimension(node->val().concate_dimension);
      concate->set_concate_num(node->srcnodes().size());
    }else if(origin=="kSplit"){
      proto.set_type(LayerT(Split));
      SplitProto *split=proto.mutable_split_conf();
      split->set_num_splits(node->dstnodes().size());
    }else if(origin=="kBridgeSrc"){
      proto.set_type(LayerT(BridgeSrc));
    }else if(origin =="kBridgeDst"){
      proto.set_type(LayerT(BridgeDst));
    }else{
      CHECK(name2layer.find(node->val().origin)!=name2layer_.end())
        <<"Unkown origin for node "<<node->val().origin;
    }
    shared_ptr<Layer> newlayer;
    if(proto.has_type()){
      // layers added due to partition
      shared_ptr<Layer> layer(factory->Create(proto.type()));
      layer->Init(proto);
      newlayer=layer;
    }else{
      // partitioned layers from origin neuralnet
      auto oldlayer=name2layer.at(node->val().origin);
      vector<int> shape=oldlayer->shape(nullptr);
      if(oldlayer->partition_type()==kNone){
        newlayer=oldlayer;
      } else{
        int pdim=oldlayer->partition_dimension();
        shape[pdim]=shape[pdim]/gsize+
          ((node->val().partitionid==gsize-1)?shape[pdim]%gsize:0);
        shared_ptr<Layer> layer(factory->Create(oldlayer->type()));
        layer->Init(*oldlayer, shape);
        layer->set_name(node->name());
        newlayer=layer;
        if(oldlayer->partition_type()==kDataPartition)
          share_conf_layers[node->val().origin].push_back(newlayer);
      }
      newlayer->set_partitionid(node->val().partitionid);
    }
    layers_.push_back(newlayer);
    name2layer_[node->name()]=newlayer;
  }

  // connect Layers.
  for(SNode node: graph_.nodes()){
    auto layer=name2layer_[node->name()];
    layer->ClearDstLayers();
    for(SNode dst: node->dstnodes())
      layer->AddDstLayer(name2layer_[dst->name()]);
    layer->ClearSrcLayers();
    for(SNode src: node->srcnodes())
      layer->AddSrcLayer(name2layer_[src->name()]);
  }

  LOG(INFO)<<"Adjacency matrix\n"<<ToAdjacency();

  // set up layers after
  int paramid=0;
  for(shared_ptr<Layer> layer: layers_){
    const vector<int>& shape=layer->shape(nullptr);
    layer->SetupAfterPartition();
    for(auto param: layer->GetParams())
      param->set_id(paramid++);
    const vector<int>& newshape=layer->shape(nullptr);
    if(shape.size())
      CHECK(std::equal(shape.begin(),shape.end(),newshape.begin()));
  }

  // share Params for layers generated from the same origin layer due to
  // data partition
  for(auto & entry: share_conf_layers){
    auto layers= entry.second;
    auto owner=layers.begin();
    auto owner_confs=(*owner)->GetParams();
    for(auto it=owner+1; it!=layers.end();it++){
      auto params=(*it)->GetParams();
      CHECK_EQ(params.size(), owner_confs.size());
      for(size_t i=0;i<params.size();i++)
        params.at(i)->ShareData(owner_confs.at(i));
    }
  }
  LOG(INFO)<<"network graph after partition layers\n"<<ToString();
}

Graph NeuralNet::CreatePartitonedGraph(const vector<shared_ptr<Layer>>& layers,
    const map<string, shared_ptr<Layer>>& name2layer){
  Graph graph;
  // partition origin nodes/layers
  map<string, vector<SNode>> layer2nodes; //from name of original layer to nodes
  int gsize=group_size_;
  for(const auto& layer: layers){
    vector<SNode> nodes;
    if(layer->partition_type()==kDataPartition||
        layer->partition_type()==kLayerPartition){
      char suffix[4];
      for(int i=0;i<gsize;i++){
        sprintf(suffix, "%02d", i);
        // differentiate partitions
        string nodename=layer->name()+"@"+string(suffix);
        auto node=graph.AddNode(nodename, LayerInfo{layer->name(), i,-1,-1});
        nodes.push_back(node);
      }
    }else if(layer->partition_type()==kNone){
      auto node=graph.AddNode(layer->name(),
          LayerInfo{layer->name(), 0,-1,-1});
      nodes.push_back(node);
    }else{
      LOG(FATAL)<<"Unknown partition type "<<layer->partition_type();
    }
    layer2nodes[layer->name()]=nodes;
  }

  // connect nodes, nodes for ConcateLayer and SliceLayer are added.
  for(shared_ptr<Layer> layer: layers){
    string name=layer->name();
    PartitionType type=layer->partition_type();
    const vector<SNode>& nodes=layer2nodes.at(name);
    for(int srcid=0;srcid<layer->srclayers_size();srcid++){
      shared_ptr<Layer> srclayer=layer->srclayers()[srcid];
      string srcname=srclayer->name();
      const vector<SNode> srcnodes=layer2nodes.at(srcname);
      PartitionType srctype=srclayer->partition_type();
      ConnectionType connection=layer->connection_type(srcid);
      if(srctype==kNone){
        CHECK_EQ(srcnodes.size(),1)
          <<"local layer "<<srcname<<" should not be partitioned";
        SNode srcnode=srcnodes[0];
        if(type==kDataPartition||(type==kLayerPartition&&connection==kOneToOne)){
          LayerInfo info=srcnode->val();
          info.slice_dimension=name2layer.at(name)->partition_dimension();
          graph.InsertSliceNode(srcnode, nodes, info);
        } else if(type==kNone){
          CHECK_EQ(nodes.size(),1)
            <<"local layer "<<name<<" should not be nodeed";
          graph.AddEdge(srcnode, nodes[0]);
        } else { // type==kLayerPartition&&connection==kOneToAll
          graph.InsertSplitNode(srcnode, nodes);
        }
      }else if((type==kNone
                &&(srctype==kDataPartition||srctype==kLayerPartition))
               ||(type==kLayerPartition&&connection==kOneToAll&&
                  (srctype==kDataPartition||srctype==kLayerPartition))){
        // copy/concate the whole srclayer for every dst partition
        for(SNode node:nodes){
          LayerInfo info=node->val();
          info.concate_dimension=name2layer.at(srcname)->partition_dimension();
          CHECK_GE(info.concate_dimension,0);
          graph.InsertConcateNode(srcnodes, node, info);
        }
      }else if((srctype==kLayerPartition&&type==kDataPartition)
          || (srctype==kDataPartition&&type==kLayerPartition)){
        // the most complext scenario
        vector<SNode> slicenodes;
        for(SNode srcnode: srcnodes){
          LayerInfo info=srcnode->val();
          info.slice_dimension=name2layer.at(name)->partition_dimension();
          slicenodes.push_back(graph.InsertSliceNode(srcnode, nodes,
              info, false));
        }
        for(SNode node: nodes){
          LayerInfo info=node->val();
          info.concate_dimension=name2layer.at(srcname)->partition_dimension();
          CHECK_GE(info.concate_dimension,0);
          graph.InsertConcateNode(slicenodes, node, info);
        }
      }else if((srctype==kDataPartition&&type==kDataPartition)||
          (srctype==kLayerPartition&&type==kLayerPartition&&
           layer->connection_type(srcid)==kOneToOne)){
        CHECK_EQ(srcnodes.size(), nodes.size());
        for(size_t i=0;i<srcnodes.size();i++){
          graph.AddEdge(srcnodes[i], nodes[i]);
        }
      }
    }
  }
  // must do topology sort, because we have added new nodes.
  graph.Sort();
  //LOG(ERROR)<<graph.ToString();

  // add node for split layer
  bool data_node=true;
  vector<SNode> oldnodes=graph.nodes();
  for(SNode node: oldnodes){
    if(node->dstnodes_size()>1&&node->val().origin!="kSlice"
        &&node->val().origin!="kSplit"&&!data_node){
      vector<SNode> dstnodes=node->dstnodes();
      for(SNode dst: dstnodes)
        graph.RemoveEdge(node, dst);
      graph.InsertSplitNode(node, dstnodes);
    }
    data_node=false;
  }

  // add bridge
  oldnodes=graph.nodes();
  for(SNode node: oldnodes){
    vector<SNode> dstnodes=node->dstnodes();
    for(size_t i=0;i<dstnodes.size();i++){
      SNode dstnode=dstnodes.at(i);
      if(node->val().partitionid!=dstnode->val().partitionid){
        graph.RemoveEdge(node, dstnode);
        graph.InsertBridgeNode(node, dstnode);
      }
    }
  }
  graph.Sort();
  return graph;
}

std::string NeuralNet::ToString(){
  map<string, string> info;
  for(auto layer: layers_){
    info[layer->name()]=IntVecToString(layer->shape(nullptr));
  }
  return graph_.ToString(info);
}

std::string NeuralNet::ToAdjacency(){
  string disp="";
  for(auto& layer: layers_){
    disp+=layer->name()+": ";
    for(const auto& dst: layer->dstlayers())
      disp+=dst->name()+", ";
    disp+="\n";
  }
  return disp;
}


void NeuralNet::ToProto(NetProto *proto, bool copyData) {
  proto->clear_layer();
}

string NeuralNet::DebugInfo(){
  string ret;
  char display[4096];
  for(auto& layer: layers_){
    if(!layer->is_datalayer()){
      sprintf(display, "Forward layer  %10s data norm1 %13.9f\n",
          layer->name().c_str(), layer->data(nullptr).asum_data());
      ret+=string(display);
    }
  }
  for (auto it = layers_.rbegin(); it != layers_.rend(); it++){
    shared_ptr<Layer> layer=*it;
    if(!(layer->is_datalayer()||layer->is_losslayer()||layer->is_parserlayer())){
      sprintf(display, "Backward layer %10s grad norm1 %13.9f\n",
          layer->name().c_str(), layer->grad(nullptr).asum_data());
      ret+=string(display);
    }
  }
  for(auto& layer: layers_){
    for(auto param: layer->GetParams()){
      sprintf(display, "Layer %10s, param id %2d, name %10s,\
          value norm1 %13.9f, grad norm1 %13.9f\n",
          layer->name().c_str(), param->id(), param->name().c_str(),
          param->data().asum_data(), param->grad().asum_data());
      ret+=string(display);
    }
  }
  return ret;
}
void NeuralNet::ShareParams(shared_ptr<NeuralNet> other, int flag){
  for(auto& layer: layers_){
    auto otherlayer=other->name2layer(layer->name());
    if(otherlayer!=nullptr){
      const auto& otherparams=otherlayer->GetParams();
      const auto& params=layer->GetParams();
      CHECK_EQ(params.size(), otherparams.size());
      for(size_t i=0;i<params.size();i++){
        params[i]->ShareData(otherparams[i]);
      }
    }
  }
}

}  // namespace singa
