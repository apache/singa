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

#ifndef SINGA_IO_ENCODER_H_
#define SINGA_IO_ENCODER_H_

#include <string>
#include <vector>

#include "singa/core/tensor.h"
#include "singa/proto/io.pb.h"

namespace singa {

/// Base encoder class that convert a set of tensors into string for storage.
class Encoder {
 public:
  Encoder() {}
  virtual ~Encoder() {}

  virtual void Setup(const EncoderConf& conf) {}

  /// Format each sample data as a string,
  /// whose structure depends on the proto definition.
  virtual std::string Encode(vector<Tensor>& data) = 0;
};

#ifdef USE_OPENCV
/// Convert an image and its label into an ImageRecord (protobuf message).
class JPGEncoder : public Encoder {
 public:
  void Setup(const EncoderConf& conf) override {
    image_dim_order_ = conf.image_dim_order();
  }
  /// 'data' has two tesors, one for the image pixels (3D) and one for the
  /// label. The image tensor's data type is kUChar.
  /// The dimension order is indicated in the EncoderConf, i.e. image_dim_order.
  /// The label tensor's data type is kInt.
  std::string Encode(vector<Tensor>& data) override;

  const std::string image_dim_order() const { return image_dim_order_; }

 private:
  /// Indicate the input image tensor's dimension order.
  std::string image_dim_order_ = "CHW";
};
#endif  // USE_OPENCV

/// Convert values from tensors into a csv formated string.
class CSVEncoder : public Encoder {
 public:
  void Setup(const EncoderConf& conf) override {}
  /// 'data' has two tesors, one for the data vector (1D) and one for the
  /// label. The data tensor's data type is kFloat.
  /// The label tensor's data type is kInt.
  std::string Encode(vector<Tensor>& data) override;
};
}  // namespace singa
#endif  // SINGA_IO_ENCODER_H_
