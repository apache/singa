#ifndef INCLUDE_NET_LAYER_H_
#define INCLUDE_NET_LAYER_H_

#include <vector>
#include <string>
#include <map>
#include <functional>
#include <utility>
#include <memory>
#include <chrono>
#include <random>
#include <lmdb.h>

#include "proto/model.pb.h"
#include "utils/data_shard.h"
#include "neuralnet/base_layer.h"


/**
 * \file this file includes the declarations neuron layer classes that conduct
 * the transformation of features.
 */
namespace singa {

/**
 * Convolution layer.
 */
class ConvolutionLayer: public Layer {
 public:
  virtual void Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers);

  /**
   * need to reset some properties (e.g., weight matrix) according to
   * shapes (after partition, e.g., partition is done against channel dimension)
   */
  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers);

  virtual void ComputeFeature(bool training, const vector<shared_ptr<Layer>>& srclayers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& srclayers);
  virtual vector<shared_ptr<Param>> GetParams() {
    return vector<shared_ptr<Param>>{weight_, bias_};
  }
  virtual ConnectionType connection_type(int k) const {
    CHECK_LT(k, srclayers_.size());
    return kOneToAll;
  }
 protected:
  int kernel_, pad_,  stride_ ;
  int batchsize_,  channels_, height_,width_;
  int col_height_, col_width_, conv_height_, conv_width_, num_filters_;
  shared_ptr<Param> weight_, bias_;
  Blob<float> col_data_, col_grad_;
};

class DropoutLayer: public Layer {
 public:
  virtual void Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers);

  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers);

  virtual void ComputeFeature(bool training, const vector<shared_ptr<Layer>>& srclayers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& srclayers);
 protected:
  // drop probability
  float pdrop_;
  /* record which neuron is dropped, required for back propagating gradients,
   * if mask[i]=0, then the i-th neuron is dropped.
   */
  Blob<float> mask_;
};

/**
  * fully connected layer
  */
class InnerProductLayer: public Layer {
 public:
  virtual void Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers);

  /**
   * need to reset weight matrix in case of LayerPartition
   */
  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers);
  virtual ConnectionType connection_type(int k) const {
    CHECK_LT(k, srclayers_.size());
    return kOneToAll;
  }

  virtual void ComputeFeature(bool training, const vector<shared_ptr<Layer>>& srclayers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& srclayers);
  //virtual void ToProto(LayerProto *layer_proto, bool copyData);
  virtual vector<shared_ptr<Param>> GetParams() {
    return vector<shared_ptr<Param>>{weight_, bias_};
  }

 private:
  //! dimension of the hidden layer
  int hdim_;
  //! dimension of the visible layer
  int vdim_;
  int batchsize_;
  shared_ptr<Param> weight_, bias_;
};

class LabelLayer: public ParserLayer {
 public:
  virtual void Setup(const LayerProto& proto, const vector<SLayer>& srclayers);
  virtual void ParseRecords(bool training, const vector<Record>& records,
      Blob<float>* blob);
};

class LRNLayer: public Layer {
/**
 * Local Response Normalization edge
 * b_i=a_i/x_i^beta
 * x_i=knorm+alpha*\sum_{j=max(0,i-n/2}^{min(N,i+n/2}(a_j)^2
 * n is size of local response area.
 * a_i, the activation (after ReLU) of a neuron convolved with the i-th kernel.
 * b_i, the neuron after normalization, N is the total num of kernels
 */

 public:
  virtual void Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers);

  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers);


  virtual void ComputeFeature(bool training, const vector<shared_ptr<Layer>>& srclayers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& srclayers);
 protected:
  //! shape of the bottom layer feature
  int batchsize_, channels_, height_, width_;
  //! size local response (neighbor) area
  int lsize_;
  //! hyper-parameter
  float alpha_, beta_, knorm_;
  Blob<float> norm_;
};

class MnistImageLayer: public ParserLayer {
 public:
  virtual void Setup(const LayerProto& proto, const vector<SLayer>& srclayers);
  virtual void ParseRecords(bool training, const vector<Record>& records,
      Blob<float>* blob);

 protected:
  // height and width of the image after deformation
  // kernel size for elastic distortion
  // n^2 images are processed as a batch for elastic distortion
  // conv height and conv width
  // gauss kernel values, displacements, column image and tmp buffer
  //float* gauss_, *displacementx_, *displacementy_, *colimg_, *tmpimg_;
  float  gamma_, beta_, sigma_, kernel_, alpha_, norm_a_, norm_b_;
  int resize_, elastic_freq_;
};

class PoolingLayer: public Layer {
 public:
  virtual void Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers);

  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers);


  virtual void ComputeFeature(bool training, const vector<shared_ptr<Layer>>& srclayers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& srclayers);
 protected:
  int kernel_, pad_, stride_;
  int batchsize_,channels_, height_, width_, pooled_height_, pooled_width_;
  PoolingProto_PoolMethod pool_;
};

class ReLULayer: public Layer {
 public:
  virtual void Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers);

  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers);


  virtual void ComputeFeature(bool training, const vector<shared_ptr<Layer>>& srclayers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& srclayers);
};


class SoftmaxLossLayer: public LossLayer {
  /*
   * connected from the label layer and the last fc layer
   */
 public:
  virtual void Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers);

  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers);
  /**
   * softmax is not recommendeded for partition because it requires the whole
   * src layer for normalization.
   */
  virtual PartitionType partition_type() const {
    if(layer_proto_.partition_type()==kLayerPartition)
      return kNone;
    else
      return layer_proto_.partition_type();
  }
  virtual ConnectionType connection_type(int k) const {
    CHECK_LT(k, srclayers_.size());
    return kOneToAll;
  }

  virtual void ComputeFeature(bool training, const vector<shared_ptr<Layer>>& srclayers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& srclayers);
 private:
  int batchsize_;
  int dim_;
  float scale_;
  int topk_;
};

class RGBImageLayer: public ParserLayer {
 public:
  virtual void Setup(const LayerProto& proto, const vector<SLayer>& srclayers);
  virtual void ParseRecords(bool training, const vector<Record>& records,
      Blob<float>* blob);

 private:
  float scale_;
  int cropsize_;
  bool mirror_;
  Blob<float> mean_;
};

class ShardDataLayer: public DataLayer{
 public:
  virtual void ComputeFeature(bool training, const vector<shared_ptr<Layer>>& srclayers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& srclayers){};
  virtual void Setup(const LayerProto& proto, const vector<SLayer>& srclayers);
 private:
  shared_ptr<DataShard> shard_;
};
class LMDBDataLayer: public DataLayer{
 public:
  virtual void ComputeFeature(bool training, const vector<shared_ptr<Layer>>& srclayers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& srclayers){};
  virtual void Setup(const LayerProto& proto, const vector<SLayer>& srclayers);
  void ConvertDatumToSingleLableImageRecord(const Datum& datum,
    SingleLabelImageRecord* record);

 private:
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
};

/**
 * This layer apply Tan function to neuron activations.
 * f(x)=A tanh(Bx)
 * f'(x)=B/A (A*A-f(x)*f(x))
 */
class TanhLayer: public Layer {
 public:
  virtual void Setup(const LayerProto& proto,
      const vector<SLayer>& srclayers);

  virtual void SetupAfterPartition(const LayerProto& proto,
      const vector<int> &shape,
      const vector<SLayer>& srclayers);


  virtual void ComputeFeature(bool training, const vector<shared_ptr<Layer>>& srclayers);
  virtual void ComputeGradient(const vector<shared_ptr<Layer>>& srclayers);
 private:
  float outer_scale_, inner_scale_;
};


}  // namespace singa

#endif  // INCLUDE_NET_LAYER_H_
