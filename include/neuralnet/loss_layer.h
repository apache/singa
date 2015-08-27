#ifndef SINGA_NEURALNET_LOSS_LAYER_H_
#define SINGA_NEURALNET_LOSS_LAYER_H_

#include "neuralnet/layer.h"

/**
 * \file this file includes the declarations of layers that inherit the base
 * LossLayer for measuring the objective training loss.
 */
namespace singa {
/**
 * Squared Euclidean loss as 0.5 ||predict - ground_truth||^2.
 */
class EuclideanLossLayer : public LossLayer {
 public:
  void ComputeFeature(int flag, Metric* perf) override;
  void ComputeGradient(int flag, Metric* perf) override;
};

/**
 * Cross-entropy loss applied to the probabilities after Softmax.
 */
class SoftmaxLossLayer : public LossLayer {
 public:
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric* perf) override;
  void ComputeGradient(int flag, Metric* perf) override;

  /**
   * softmax is not recommendeded for partition because it requires the whole
   * src layer for normalization.
   */
  ConnectionType src_neuron_connection(int k) const override {
    // CHECK_LT(k, srclayers_.size());
    return kOneToAll;
  }

 private:
  int batchsize_;
  int dim_;
  float scale_;
  int topk_;
};
}
//  namespace singa
#endif  // SINGA_NEURALNET_LOSS_LAYER_H_
