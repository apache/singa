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

#include "singa/io/decoder.h"
#include <string>
#include <sstream>

#define MAXSIZE 4096

namespace singa {

std::vector<Tensor> TextDecoder::Decode(std::string value) {
  std::vector<Tensor> output;
  std::stringstream ss;
  ss.str(value);
  int l = 0;
  if (has_label_ == true) ss >> l;
  std::string str;
  float* d = new float[MAXSIZE];
  size_t size = 0;
  while(std::getline(ss, str, ',')) {
    float temp;
    if (std::stringstream(str) >> temp) {
      CHECK_LE(size, MAXSIZE-1);
      d[size++] = temp;
    }
  }

  Tensor data(Shape{size}, kFloat32);
  data.CopyDataFromHostPtr(d, size);
  output.push_back(data);
  if (has_label_ == true) {
    Tensor label(Shape{1}, kInt);
    label.CopyDataFromHostPtr(&l, 1);
    output.push_back(label);
  }
  return output;
}
}  // namespace singa
