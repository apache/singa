#ifndef SINGA_NEURALNET_NEURALNET_H_
#define SINGA_NEURALNET_NEURALNET_H_

#include <vector>
#include <map>
#include <memory>
#include <string>

#include "proto/job.pb.h"
#include "neuralnet/layer.h"
#include "utils/factory.h"
#include "utils/graph.h"

namespace singa {
using std::vector;
using std::string;
using std::map;
using std::shared_ptr;

/**
 * The neural network is constructed from user configurations in NetProto.
 *
 * Some layers, e.g., SplitLayer and BridgeSrcLayer/BridgeDstLayer
 * will be added implicitly to partition the neural network.
 * TODO create wrappers for popular models, e.g., MLP, CNN.
 */
class NeuralNet {
 public:
  /**
   * Create the neural network for training, test or validation.
   *
   * Parameters for test/validation net can share those from training after
   * setup (done outside of this funcion).
   *
   * @param np proto for the neural network
   * @param phase test/training/validation
   * @param num num of partitions, do partitioning if num > 1
   * @return shared pointer to a neural net
   */
  static shared_ptr<NeuralNet> Create(const NetProto& np, Phase phase, int num);

 public:
  /**
   * construct the net structure from protocol buffer.
   * @param netproto neural net config
   * @param npartitions num of partitions. 1 for no partitioning.
   */
  explicit NeuralNet(NetProto netproto, int npartitions = 1);
  ~NeuralNet();
  /**
   * To display the adjacency layers
   */
  std::string ToAdjacency();
  /**
   * Share memory of parameter values from other neuralnet
   */
  void ShareParamsFrom(shared_ptr<NeuralNet> other);

  const std::vector<Layer*>& layers() {
    return layers_;
  }
  const std::vector<ParserLayer*>& parserlayers() const {
    LOG(FATAL)<< " not implemented";
    return parserlayers_;
  }
  const std::vector<LossLayer*>& losslayers() const {
    LOG(FATAL)<< " not implemented";
    return losslayers_;
  }
  const std::vector<DataLayer*>& datalayers() const {
    LOG(FATAL)<< " not implemented";
    return datalayers_;
  }
  const std::vector<Param*>& params() const {
    return params_;
  }
  Layer* name2layer(string name) const {
    if (name2layer_.find(name) != name2layer_.end())
      return name2layer_.at(name);
    else
      return nullptr;
  }
  Param* paramid2param(int id) const {
    return paramid2param_.at(id);
  }

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
  vector<Layer*> layers_;
  vector<ParserLayer*> parserlayers_;
  vector<LossLayer*> losslayers_;
  vector<DataLayer*> datalayers_;
  vector<Param*> params_;

  map<string, Layer*> name2layer_;
  map<int, Param*> paramid2param_;
};
}  // namespace singa
#endif  // SINGA_NEURALNET_NEURALNET_H_
