#include "singa.h"
#include "rnnlm.pb.h"
namespace singa {

/**
 * Base RNN layer. May make it a base layer of SINGA.
 */
class RNNLayer : public NeuronLayer {
 public:
  /**
   * The recurrent layers may be unrolled different times for different
   * iterations, depending on the applications. For example, the ending word
   * of a sentence may stop the unrolling; unrolling also stops when the max
   * window size is reached. Every layer must reset window_ in its
   * ComputeFeature function.
   *
   * @return the effective BPTT length, which is <= max_window.
   */
  inline int window() { return window_; }

 protected:
  //!< effect window size for BPTT
  int window_;
};

/**
 * Input layer that get read records from data shard
 */
class RnnDataLayer : public RNNLayer {
 public:
  ~RnnDataLayer();
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric *perf) override;
  void ComputeGradient(int flag, Metric* perf) override {}
  int max_window() const {
    return max_window_;
  }

  const std::vector<singa::WordRecord>& records() const {
    return records_;
  }

 private:
  int max_window_;
  DataShard* shard_;
  std::vector<singa::WordRecord> records_;
};


/**
 * WordLayer that read records_[0] to records_[window_ - 1] from RnnDataLayer to offer data for computation
 */
class WordLayer : public RNNLayer {
 public:
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric *perf) override;
  void ComputeGradient(int flag, Metric* perf) override {}
};


/**
 * LabelLayer that read records_[1] to records_[window_] from RnnDataLayer to offer label information
 */
class RnnLabelLayer : public RNNLayer {
 public:
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric *perf) override;
  void ComputeGradient(int flag, Metric* perf) override {}
};


/**
 * Word embedding layer that get one row from the embedding matrix for each
 * word based on the word index
 */
class EmbeddingLayer : public RNNLayer {
 public:
  ~EmbeddingLayer();
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric *perf) override;
  void ComputeGradient(int flag, Metric* perf) override;
  const std::vector<Param*> GetParams() const override {
    std::vector<Param*> params{embed_};
    return params;
  }


 private:
  int word_dim_;
  int vocab_size_;
  //!< word embedding matrix of size vocab_size_ x word_dim_
  Param* embed_;
};


/**
 * hid[t] = sigmoid(hid[t-1] * W + src[t])
 */
class HiddenLayer : public RNNLayer {
 public:
  ~HiddenLayer();
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric *perf) override;
  void ComputeGradient(int flag, Metric* perf) override;
  const std::vector<Param*> GetParams() const override {
    std::vector<Param*> params{weight_};
    return params;
  }


 private:
  Param* weight_;
};

/**
 * p(word at t+1 is from class c) = softmax(src[t]*Wc)[c]
 * p(w|c) = softmax(src[t]*Ww[Start(c):End(c)])
 * p(word at t+1 is w)=p(word at t+1 is from class c)*p(w|c)
 */
class OutputLayer : public RNNLayer {
 public:
  ~OutputLayer();
  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric *perf) override;
  void ComputeGradient(int flag, Metric* perf) override;
  const std::vector<Param*> GetParams() const override {
    std::vector<Param*> params{word_weight_, class_weight_};
    return params;
  }

 private:
  vector<Blob<float>> pword_;
  Blob<float> pclass_;
  Param* word_weight_, *class_weight_;
};
}
