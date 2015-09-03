#ifndef SINGA_NEURALNET_NEURON_LAYER_H_
#define SINGA_NEURALNET_NEURON_LAYER_H_
#include <vector>

#include "neuralnet/layer.h"
#include "proto/job.pb.h"
/**
 * \file this file includes the declarations neuron layer classes that conduct
 * the transformation of features.
 */
namespace singa {
/**
 * Convolution layer.
 */
class ConvolutionLayer : public NeuronLayer {
 public:
  ~ConvolutionLayer();

  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric* perf) override;
  void ComputeGradient(int flag, Metric* perf) override;
  const std::vector<Param*> GetParams() const override {
    std::vector<Param*> params{weight_, bias_};
    return params;
  }
  ConnectionType src_neuron_connection(int k) const  override {
    // CHECK_LT(k, srclayers_.size());
    return kOneToAll;
  }

 protected:
  int kernel_, pad_,  stride_;
  int batchsize_,  channels_, height_, width_;
  int col_height_, col_width_, conv_height_, conv_width_, num_filters_;
  Param* weight_, *bias_;
  Blob<float> col_data_, col_grad_;
};

class DropoutLayer : public NeuronLayer {
 public:
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric* perf) override;
  void ComputeGradient(int flag, Metric* perf) override;
 protected:
  // drop probability
  float pdrop_;
  /* record which neuron is dropped, required for back propagating gradients,
   * if mask[i]=0, then the i-th neuron is dropped.
   */
  Blob<float> mask_;
};
/**
 * Local Response Normalization edge
 *
 * b_i=a_i/x_i^beta
 * x_i=knorm+alpha*\sum_{j=max(0,i-n/2}^{min(N,i+n/2}(a_j)^2
 * n is size of local response area.
 * a_i, the activation (after ReLU) of a neuron convolved with the i-th kernel.
 * b_i, the neuron after normalization, N is the total num of kernels
 */
class LRNLayer : public NeuronLayer {
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric *perf) override;
  void ComputeGradient(int flag, Metric* perf) override;

 protected:
  //! shape of the bottom layer feature
  int batchsize_, channels_, height_, width_;
  //! size local response (neighbor) area
  int lsize_;
  //! hyper-parameter
  float alpha_, beta_, knorm_;
  Blob<float> norm_;
};

class PoolingLayer : public NeuronLayer {
 public:
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric *perf) override;
  void ComputeGradient(int flag, Metric* perf) override;

 protected:
  int kernel_, pad_, stride_;
  int batchsize_, channels_, height_, width_, pooled_height_, pooled_width_;
  PoolingProto_PoolMethod pool_;
};

class ReLULayer : public NeuronLayer {
 public:
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric *perf) override;
  void ComputeGradient(int flag, Metric* perf) override;
};

class InnerProductLayer : public NeuronLayer {
 public:
  ~InnerProductLayer();
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric* perf) override;
  void ComputeGradient(int flag, Metric* perf) override;
  const std::vector<Param*> GetParams() const override {
    std::vector<Param*> params{weight_, bias_};
    return params;
  }

 private:
  int batchsize_;
  int vdim_, hdim_;
  bool transpose_;
  Param *weight_, *bias_;
};

/**
 * This layer apply scaled Tan function to neuron activations.
 * f(x)=1.7159047  tanh(0.66666667 x)
 */
class STanhLayer : public NeuronLayer {
 public:
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric *perf) override;
  void ComputeGradient(int flag, Metric* perf) override;
};

/**
 * This layer apply Sigmoid function to neuron activations.
 * f(x)=1/(1+exp(-x))
 * f'(x)=f(x)*(1-f(x))
 */
class SigmoidLayer: public Layer {
 public:
  using Layer::ComputeFeature;
  using Layer::ComputeGradient;

  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric* perf) override;
  void ComputeGradient(int flag, Metric* perf) override;
};


/**
 * Base layer for RBM models.
 */
class RBMLayer: public Layer {
 public:
  virtual ~RBMLayer() {}
  void Setup(const LayerProto& proto, int npartitions) override;
  const Blob<float>& neg_data(const Layer* layer) {
    return neg_data_;
  }
  Blob<float>* mutable_neg_data(const Layer* layer) {
    return &neg_data_;
  }
  const std::vector<Param*> GetParams() const override {
    std::vector<Param*> params{weight_, bias_};
    return params;
  }
  virtual Blob<float>* Sample(int flat);

 protected:
  //! if ture, sampling according to guassian distribution
  bool gaussian_;
  //! dimension of the hidden layer
  int hdim_;
  //! dimension of the visible layer
  int vdim_;
  int batchsize_;
  bool first_gibbs_;
  Param* weight_, *bias_;

  Blob<float> neg_data_;
  Blob<float> neg_sample_;
  Blob<float> sample_;
};

/**
 * RBM visible layer
 */
class RBMVisLayer: public RBMLayer {
 public:
  ~RBMVisLayer();
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric* perf) override;
  void ComputeGradient(int flag, Metric* perf) override;

 private:
  RBMLayer* hid_layer_;
  Layer* input_layer_;
};
/**
 * RBM hidden layer
 */
class RBMHidLayer: public RBMLayer {
 public:
  ~RBMHidLayer();
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric* perf) override;
  void ComputeGradient(int flag, Metric* perf) override;

 private:
  RBMLayer *vis_layer_;
};


}  // namespace singa

#endif  // SINGA_NEURALNET_NEURON_LAYER_H_
