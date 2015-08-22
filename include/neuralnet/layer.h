#ifndef SINGA_NEURALNET_LAYER_H_
#define SINGA_NEURALNET_LAYER_H_

#include <vector>
#include "neuralnet/base_layer.h"
#include "proto/job.pb.h"
#include "utils/data_shard.h"

/**
 * \file this file includes the declarations neuron layer classes that conduct
 * the transformation of features.
 */
namespace singa {

/********** Derived from DataLayer **********/

class ShardDataLayer : public DataLayer {
 public:
  ~ShardDataLayer();

  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(Phase phase, Metric *perf) override;

 private:
  DataShard* shard_;
};

/********** Derived from ParserLayer **********/

class LabelLayer : public ParserLayer {
 public:
  void Setup(const LayerProto& proto, int npartitions) override;
  void ParseRecords(Phase phase, const std::vector<Record>& records,
                    Blob<float>* blob) override;
};

class MnistLayer : public ParserLayer {
 public:
  void Setup(const LayerProto& proto, int npartitions) override;
  void ParseRecords(Phase phase, const std::vector<Record>& records,
                    Blob<float>* blob) override;

 protected:
  // height and width of the image after deformation
  // kernel size for elastic distortion
  // n^2 images are processed as a batch for elastic distortion
  // conv height and conv width
  // gauss kernel values, displacements, column image and tmp buffer
  // float* gauss_, *displacementx_, *displacementy_, *colimg_, *tmpimg_;
  float  gamma_, beta_, sigma_, kernel_, alpha_, norm_a_, norm_b_;
  int resize_, elastic_freq_;
};

class RGBImageLayer : public ParserLayer {
 public:
  void Setup(const LayerProto& proto, int npartitions) override;
  void ParseRecords(Phase phase, const std::vector<Record>& records,
                    Blob<float>* blob) override;

 private:
  float scale_;
  int cropsize_;
  bool mirror_;
  Blob<float> mean_;
};

/********** Derived from NeuronLayer **********/

/**
 * Convolution layer.
 */
class ConvolutionLayer : public NeuronLayer {
 public:
  ~ConvolutionLayer();

  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric *perf) override;
  void ComputeGradient(int flag) override;
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
 * RBM visible layer
 */
class RBMVisLayer: public RBMLayer {
 public:
  using Layer::ComputeFeature;
  using Layer::ComputeGradient;

  ~RBMVisLayer();
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric *perf) override;
  void ComputeGradient(int flag) override;
  Blob<float>* Sample(int flat) override;

 private:
  RBMLayer* hid_layer_;
  Layer* input_layer_;
};

class LRNLayer : public NeuronLayer {
/**
 * Local Response Normalization edge
 * b_i=a_i/x_i^beta
 * x_i=knorm+alpha*\sum_{j=max(0,i-n/2}^{min(N,i+n/2}(a_j)^2
 * n is size of local response area.
 * a_i, the activation (after ReLU) of a neuron convolved with the i-th kernel.
 * b_i, the neuron after normalization, N is the total num of kernels
 */
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(Phase phase, Metric *perf) override;
  void ComputeGradient(Phase phase, Metric* perf) override;

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
  void ComputeFeature(Phase phase, Metric *perf) override;
  void ComputeGradient(Phase phase, Metric* perf) override;

 protected:
  int kernel_, pad_, stride_;
  int batchsize_, channels_, height_, width_, pooled_height_, pooled_width_;
  PoolingProto_PoolMethod pool_;
};

class ReLULayer : public NeuronLayer {
 public:
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(Phase phase, Metric *perf) override;
  void ComputeGradient(Phase phase, Metric* perf) override;
};

/**
 * RBM hidden layer
 */
class RBMHidLayer: public RBMLayer {
 public:
  ~RBMHidLayer();

  ~RBMHidLayer();
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric* perf) override;
  void ComputeGradient(int flag, Metric* perf) override;
  Blob<float>* Sample(int flat) override;
 private:
  // whether use gaussian sampling
  bool gaussian_;
  RBMLayer *vis_layer_;
};

/**
  * RBM visible layer
  */
class RBMVisLayer : public NeuronLayer {
 public:
  ~RBMVisLayer();

  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric* perf) override;
  void ComputeGradient(int flag, Metric* perf) override;

  ConnectionType src_neuron_connection(int k) const override {
    // CHECK_LT(k, srclayers_.size());
    return kOneToAll;
  }
  const Blob<float>& data(const Layer* from, Phase phase) const override {
    return (phase == kPositive) ? data_ : vis_sample_;
  }
  const std::vector<Param*> GetParams() const override {
    std::vector<Param*> params{weight_, bias_};
    return params;
  }

 private:
  //! dimension of the hidden layer
  int hdim_;
  //! dimension of the visible layer
  int vdim_;
  int batchsize_;
  bool transpose_;
  Param* weight_, *bias_;
  // data to store sampling result
  Blob<float> vis_sample_;
  // in order to implement Persistent Contrastive Divergence,
};

/**
 * This layer apply Tan function to neuron activations.
 * f(x)=A tanh(Bx)
 * f'(x)=B/A (A*A-f(x)*f(x))
 */
class TanhLayer : public NeuronLayer {
 public:
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(Phase phase, Metric *perf) override;
  void ComputeGradient(Phase phase, Metric* perf) override;

 private:
  float outer_scale_, inner_scale_;
};

/********** Derived from LossLayer **********/

class SoftmaxLossLayer : public LossLayer {
  /*
   * connected from the label layer and the last fc layer
   */
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

/********** Derived from BridgeLayer **********/

/**
 * For recv data from layer on other threads which may resident on other nodes
 * due to layer/data partiton
 */
class BridgeDstLayer : public BridgeLayer {
 public:
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric* perf) override {
    // reset ready_ for next iteration.
    ready_ = false;
  }
  void ComputeGradient(int flag, Metric* perf) override {}
  bool is_bridgedstlayer() const {
    return true;
  }
};

/**
 * For sending data to layer on other threads which may resident on other nodes
 * due to layer/data partition.
 */
class BridgeSrcLayer : public BridgeLayer {
 public:
  void ComputeFeature(Phase phase, Metric* perf) override {}
  void ComputeGradient(Phase phase, Metric* perf) override {
    ready_ = false;
  }
  const Blob<float>& data(const Layer* from, Phase phase) const override {
    return srclayers_[0]->data(this);
  }
  Blob<float>* mutable_data(const Layer* from, Phase phase) override {
    return srclayers_[0]->mutable_data(this);
  }
  const Blob<float>& grad(const Layer* from) const override {
    return srclayers_[0]->grad(this);
  }
  Blob<float>* mutable_grad(const Layer* from) override {
    return srclayers_[0]->mutable_grad(this);
  }
  bool is_bridgesrclayer() const override {
    return true;
  }
};

/********** Derived from ConnectionLayer **********/

/**
 * Concate src layers on one dimension
 */
class ConcateLayer : public ConnectionLayer {
 public:
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(Phase phase, Metric* perf) override;
  void ComputeGradient(Phase phase, Metric* perf) override;
};

/**
 * Slice the source layer into multiple dst layers on one dimension
 */
class SliceLayer : public ConnectionLayer {
 public:
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric *perf) override;

 private:
  std::vector<Blob<float>> datavec_;
  std::vector<Blob<float>> gradvec_;
  int slice_dim_;
  int slice_num_;
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
 * Connect the source layer with multiple dst layers.
 * Pass source layer's data blob directly to dst layers.
 * Aggregate dst layer's gradients into source layer's gradient.
 */
class SplitLayer : public ConnectionLayer {
 public:
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric* perf) override;
  void ComputeGradient(int flag, Metric* perf) override;

 protected:
  Blob<float> grads_;
};

}  // namespace singa

#endif  // SINGA_NEURALNET_LAYER_H_
