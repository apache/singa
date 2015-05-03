#ifndef INCLUDE_NET_NET_H_
#define INCLUDE_NET_NET_H_

#include <glog/logging.h>
#include <vector>
#include <map>
#include <memory>

#include "proto/model.pb.h"
#include "neuralnet/layer.h"
#include "utils/factory.h"
#include "utils/graph.h"

using std::vector;
using std::string;
using std::map;
using std::shared_ptr;
namespace singa {
/**
 * The neural network is constructed from user configured layers through google
 * protocol buffer. TODO support constructing neural network by adding layers
 * explicitly. E.g., users create layers and connect them manually in the code.
 *
 * Some layers, e.g., SplitLayer and BridgeSrcLayer/BridgeDstLayer will be added
 * implicitly to partition the neural network.
 */
class NeuralNet {
 public:
  /**
   * Register Layers
   */
  static void RegisterLayers();
  /**
   * Setup the neural network for training, test or validation.
   *
   * Parameters for test/validation net can share those from training after
   * setup (done outside of this funcion).
   *
   * @param np proto for the neural network.
   */
  static shared_ptr<NeuralNet> SetupNeuralNet(const NetProto& np, Phase phase);

 public:
  /**
   * construct the net structure from protocol buffer.
   */
  NeuralNet(NetProto net_proto, int group_size=1);
  /**
   * construct a json string representing the neuralnet graph.
   * The json string can be used by other graph engine to draw a figure for
   * displaying the neuralnet structure.
   */
  std::string ToString();
  /**
   * Print Norm1 of data and grad of each Layer and parameter.
   * @param net, neural network
   */
  string DebugInfo();

  /**
   * to display the adjacency layers
   */
  std::string ToAdjacency();
  /**
   * Add layer explicitly used in manually programming/constructing neural net.
   */
  void AddLayer(const LayerProto &layer_proto){};
  /**
   * Add layer explicitly used in manually programming/constructing neural net.
   */
  void AddLayer(const Layer* layer){};
  /**
   * share weights from other neuralnet
   */
  void ShareParams(shared_ptr<NeuralNet> other,int flag);
  void ToProto(NetProto *net_proto, bool copyData=false);
  const std::vector<shared_ptr<Layer>>& layers() {
    return layers_;
  }
  /**
   * return ParserLayer of the neuralnet.
   */
  const std::vector<ParserLayer*>& parserlayers() {
    if(parserlayers_.size()==0){
      for(auto& layer: layers_)
        if(layer->is_parserlayer())
          parserlayers_.push_back(static_cast<ParserLayer*>(layer.get()));
    }
    return parserlayers_;
  }
  const std::vector<LossLayer*>& losslayers() {
    if(losslayers_.size()==0){
      for(auto& layer: layers_)
        if(layer->is_losslayer())
          losslayers_.push_back(static_cast<LossLayer*>(layer.get()));
    }
    return losslayers_;
  }
  const std::vector<DataLayer*>& datalayers() {
    if(datalayers_.size()==0){
      for(auto& layer: layers_)
        if(layer->is_datalayer())
          datalayers_.push_back(static_cast<DataLayer*>(layer.get()));
    }
    return datalayers_;
  }
  const std::vector<shared_ptr<Param>> &params()const {
    return params_;
  }
  shared_ptr<Layer> name2layer(string name){
    if (name2layer_.find(name)!=name2layer_.end())
      return name2layer_[name];
    else return nullptr;
  }

  shared_ptr<Param> paramid2param(int id) {
    if(paramid2param_.size()==0){
      for(auto& layer: layers_){
        for(shared_ptr<Param> p: layer->GetParams()){
          paramid2param_[p->id()]=p;
        }
      }
    }
    return paramid2param_[id];
  }

 protected:
  void ConstructNeuralNet(const NetProto &net_proto);
  void PartitionNeuralNet();
  map<string, shared_ptr<Layer>> GetNameToLayer(
    const vector<shared_ptr<Layer>>& layers);
  Graph CreatePartitonedGraph(const vector<shared_ptr<Layer>>& layers,
    const map<string, shared_ptr<Layer>>& name2layer);

  /**
   * Partition each layer according its partition type and dimension.
   * @param layers original unpartitioned layers
   */
  map<string, vector<shared_ptr<Layer>>> PartitionLayers(
      const vector<shared_ptr<Layer>>& layers);

 protected:
  vector<shared_ptr<Layer>> layers_;
  vector<ParserLayer*> parserlayers_;
  vector<LossLayer*> losslayers_;
  vector<DataLayer*> datalayers_;
  vector<shared_ptr<Param>> params_;
  map<string, shared_ptr<Layer>> name2layer_;
  map<int, shared_ptr<Param>> paramid2param_;

  map<string, LayerProto> name2layerproto_;
  int group_size_;
  Graph graph_;
};
}  // namespace singa
#endif  // INCLUDE_NET_NET_H_
