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
#include <sstream>
#include <fstream>
#include "singa/neuralnet/input_layer.h"
namespace singa {

void CharRNNInputLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  InputLayer::Setup(conf, srclayers);
  batchsize_ = conf.char_rnn_conf().batchsize();
  path_ = conf.char_rnn_conf().path();
  vocab_path_ = conf.char_rnn_conf().vocab_path();
  unroll_len_ = conf.char_rnn_conf().unroll_len();
  datavec_.clear();
  // each unroll layer has a input blob
  for (int i = 0; i <= unroll_len_; i++) {
    datavec_.push_back(new Blob<float>(batchsize_));
  }
}

void CharRNNInputLayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  if (buf_.size() == 0) {

    // read the vocab
    {
      std::ifstream fin;
      fin.open(vocab_path_);
      CHECK(fin.is_open()) << "Can't open vocab_path = " << vocab_path_;
      std::stringstream stream;
      stream << fin.rdbuf();
      string vocab = stream.str();
      LOG(ERROR) << "Vocab_size = " << vocab.length();
      for (char c : vocab)
        char2index_[c] = char2index_.size() - 1;
      fin.close();
    }

    // read the whole text file
    {
      std::ifstream fin;
      fin.open(path_);
      CHECK(fin.is_open()) << "Can't open filepath = " << path_;
      std::stringstream stream;
      stream << fin.rdbuf();
      buf_ = stream.str();
      fin.close();
    }

    // decide the start pos of each instance in one mini-batch
    int max_offset = buf_.length() / batchsize_;
    CHECK_GT(max_offset, unroll_len_);
    LOG(ERROR) << "Max iteration per epoch = " << max_offset / unroll_len_;
    for (int i = 0; i < batchsize_; i ++) {
      start_.push_back(i * max_offset);
    }
  }

  for (int l = 0; l < unroll_len_ + 1; l++) {
    float* ptr = datavec_[l]->mutable_cpu_data();
    for (int i = 0; i < batchsize_; i++) {
      ptr[i] = static_cast<float>(char2index_.at(buf_[start_[i] + offset_ + l]));
    }
  }
  offset_ += unroll_len_;
  if (offset_ >= buf_.length() / batchsize_) {
//  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
//  std::mt19937 g(seed);
//  std::shuffle(start_.begin(), start_.end(), g);
    offset_ = 0;
    // return -1;
  }
}
}  // namespace singa
