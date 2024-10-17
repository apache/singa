/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SINGA_IO_DECODER_H_
#define SINGA_IO_DECODER_H_

#include <string>
#include <vector>

#include "singa/core/tensor.h"
#include "singa/proto/io.pb.h"

namespace singa {
/// The base decoder that converts a string into a set of tensors.
class Decoder {
 public:
  Decoder() {}
  virtual ~Decoder() {}

  virtual void Setup(const DecoderConf& conf) {}

  /// Decode value to get data and labels
  virtual std::vector<Tensor> Decode(std::string value) = 0;
};

#ifdef USE_OPENCV
/// Decode the string as an ImageRecord object and convert it into a image
/// tensor (dtype is kFloat32) and a label tensor (dtype is kInt).
class JPGDecoder : public Decoder {
 public:
  void Setup(const DecoderConf& conf) override {
    image_dim_order_ = conf.image_dim_order();
  }
  std::vector<Tensor> Decode(std::string value) override;

  const std::string image_dim_order() const { return image_dim_order_; }

 private:
  /// Indicate the dimension order for the output image tensor.
  std::string image_dim_order_ = "CHW";
};
#endif

/// Decode the string of csv formated data  into data tensor
/// (dtype is kFloat32) and optionally a label tensor (dtype is kInt).
class CSVDecoder : public Decoder {
 public:
  void Setup(const DecoderConf& conf) override {
    has_label_ = conf.has_label();
  }
  std::vector<Tensor> Decode(std::string value) override;

  bool has_label() const { return has_label_; }

 private:
  /// if ture the first value is the label
  bool has_label_ = false;
};
}  // namespace singa
#endif  // SINGA_IO_DECODER_H_
