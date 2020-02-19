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

#include <algorithm>
#include <sstream>

#include "gtest/gtest.h"
#include "singa/io/decoder.h"
#include "singa/io/encoder.h"

using singa::Shape;
using singa::Tensor;
TEST(CSV, EncoderDecode) {
  singa::CSVEncoder encoder;
  singa::CSVDecoder decoder;

  singa::DecoderConf decoder_conf;
  decoder_conf.set_has_label(true);
  decoder.Setup(decoder_conf);
  EXPECT_EQ(true, decoder.has_label());

  float in_data[] = {1.23f, 4.5f, 5.1f, 3.33f, 0.44f};
  std::string in_str = "2, 1.23, 4.5, 5.1, 3.33, 0.44";
  int in_label = 2;
  size_t size = 5;

  std::vector<Tensor> input;
  Tensor data(Shape{size}, singa::kFloat32), label(Shape{1}, singa::kInt);
  data.CopyDataFromHostPtr<float>(in_data, size);
  label.CopyDataFromHostPtr<int>(&in_label, 1);
  input.push_back(data);
  input.push_back(label);

  std::string value = encoder.Encode(input);
  in_str.erase(std::remove(in_str.begin(), in_str.end(), ' '), in_str.end());
  EXPECT_EQ(in_str, value);

  std::vector<Tensor> output = decoder.Decode(value);
  const auto* out_data = output.at(0).data<float>();
  const auto* out_label = output.at(1).data<int>();
  for (size_t i = 0; i < size; i++) EXPECT_EQ(in_data[i], out_data[i]);
  EXPECT_EQ(in_label, out_label[0]);
}
