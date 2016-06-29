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

#include "singa/io/snapshot.h"

#include <string>
#include <unordered_set>
#include <unordered_map>
#include <memory>
#include <utility>
#include <iostream>

namespace singa {
Snapshot::Snapshot(const std::string& prefix, Mode mode)
    : prefix_(prefix),
      mode_(mode),
      bin_writer_ptr_(mode_ == kWrite ? (new io::BinFileWriter) : nullptr),
      text_writer_ptr_(mode_ == kWrite ? (new io::TextFileWriter) : nullptr),
      bin_reader_ptr_(mode_ == kRead ? (new io::BinFileReader) : nullptr) {
  if (mode_ == kWrite) {
    bin_writer_ptr_->Open(prefix + ".model", io::kCreate);
    text_writer_ptr_->Open(prefix + ".desc", io::kCreate);
  } else if (mode == kRead) {
    bin_reader_ptr_->Open(prefix + ".model");
    std::string key, serialized_str;
    singa::TensorProto tp;
    while (bin_reader_ptr_->Read(&key, &serialized_str)) {
      CHECK(param_names_.count(key) == 0);
      param_names_.insert(key);
      CHECK(tp.ParseFromString(serialized_str));
      param_map_[key].FromProto(tp);
    }
  } else {
    LOG(FATAL)
        << "Mode for snapshot should be Snapshot::kWrite or Snapshot::kRead";
  }
}

void Snapshot::Write(const std::string& key, const Tensor& param) {
  CHECK(mode_ == kWrite);
  CHECK(param_names_.count(key) == 0);
  param_names_.insert(key);
  TensorProto tp;
  param.ToProto(&tp);
  std::string serialized_str;
  CHECK(tp.SerializeToString(&serialized_str));
  bin_writer_ptr_->Write(key, serialized_str);

  std::string desc_str = "parameter name: " + key;
  Shape shape = param.shape();
  desc_str += "\tdata type: " + std::to_string(param.data_type());
  desc_str += "\tdim: " + std::to_string(shape.size());
  desc_str += "\tshape:";
  for (size_t s : shape) desc_str += " " + std::to_string(s);
  text_writer_ptr_->Write(key, desc_str);
}

std::vector<std::pair<std::string, Tensor>> Snapshot::Read() {
  CHECK(mode_ == kRead);
  std::vector<std::pair<std::string, Tensor>> ret;
  for (auto it = param_map_.begin(); it != param_map_.end(); ++it)
    ret.push_back(*it);
  return ret;
}

std::vector<std::pair<std::string, Shape>> Snapshot::ReadShape() {
  CHECK(mode_ == kRead);
  std::vector<std::pair<std::string, Shape>> ret;
  for (auto it = param_map_.begin(); it != param_map_.end(); ++it)
    ret.push_back(std::make_pair(it->first, it->second.shape()));
  return ret;
}

Tensor Snapshot::Read(const std::string& key) {
  CHECK(mode_ == kRead);
  CHECK(param_map_.count(key) == 1);
  return param_map_[key];
}

Shape Snapshot::ReadShape(const std::string& key) {
  CHECK(mode_ == kRead);
  CHECK(param_map_.count(key) == 1);
  return param_map_[key].shape();
}

}  //  namespace singa
