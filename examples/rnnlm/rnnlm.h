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

#ifndef EXAMPLES_RNNLM_RNNLM_H_
#define EXAMPLES_RNNLM_RNNLM_H_

#include <string>
#include <vector>
#include "singa/singa.h"
#include "./rnnlm.pb.h"

namespace rnnlm {
using std::vector;
using singa::LayerProto;
using singa::Layer;
using singa::Param;
using singa::Blob;
using singa::Metric;
/**
 * Base RNN layer. May make it a base layer of SINGA.
 */
class RNNLayer : virtual public singa::Layer {
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
class DataLayer : public RNNLayer, public singa::InputLayer {
 public:
  ~DataLayer();
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  int max_window() const {
    return max_window_;
  }

 private:
  int max_window_;
  singa::io::Store* store_ = nullptr;
};


/**
 * LabelLayer that read records_[1] to records_[window_] from DataLayer to
 * offer label information
class LabelLayer : public RNNLayer {
 public:
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override {}
};
 */


/**
 * Word embedding layer that get one row from the embedding matrix for each
 * word based on the word index
 */
class EmbeddingLayer : public RNNLayer {
 public:
  ~EmbeddingLayer();
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;
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
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;

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
class LossLayer : public RNNLayer {
 public:
  ~LossLayer();
  void Setup(const LayerProto& conf, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override;

  const std::string ToString(bool debug, int flag) override;
  const std::vector<Param*> GetParams() const override {
    std::vector<Param*> params{word_weight_, class_weight_};
    return params;
  }

 private:
  std::vector<Blob<float>> pword_;
  Blob<float> pclass_;
  Param* word_weight_, *class_weight_;
  float loss_, ppl_;
  int num_;
};
}  // namespace rnnlm
#endif  // EXAMPLES_RNNLM_RNNLM_H_
