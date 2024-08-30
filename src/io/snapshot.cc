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

#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "singa/singa_config.h"

namespace singa {
Snapshot::Snapshot(const std::string& prefix, Mode mode,
                   int max_param_size /*in MB*/)
    : prefix_(prefix),
      mode_(mode),
      bin_writer_ptr_(mode_ == kWrite ? (new io::BinFileWriter) : nullptr),
      text_writer_ptr_(mode_ == kWrite ? (new io::TextFileWriter) : nullptr),
      bin_reader_ptr_(mode_ == kRead ? (new io::BinFileReader) : nullptr) {
  if (mode_ == kWrite) {
    // changed to .bin since v1.0.1
    bin_writer_ptr_->Open(prefix + ".bin", io::kCreate, max_param_size << 20);
    text_writer_ptr_->Open(prefix + ".desc", io::kCreate);

    // write the current version ids
    // text_writer_ptr_->Write("SINGA_VERSION", std::to_string(SINGA_VERSION));
    text_writer_ptr_->Write("",
                            "SINGA VERSION: " + std::to_string(SINGA_VERSION));
  } else if (mode == kRead) {
    /*
    auto text_reader_ptr = new io::TextFileReader();
    text_reader_ptr->Open(prefix + ".desc");
    std::string key, val;
    while (text_reader_ptr->Read(&key, &val)) {
      if (key == "0")
        version_ = std::stoi(val);
    }
    delete text_reader_ptr;
    */
    std::string key, val;
    if (!bin_reader_ptr_->Open(prefix + ".bin", max_param_size << 20))
      CHECK(bin_reader_ptr_->Open(prefix + ".model", max_param_size << 20))
          << "Cannot open the checkpoint bin file:"
          << prefix + ".bin (>=1.0.1) " << " or "
          << prefix + " .model (used by 1.0.0)";
    singa::TensorProto tp;
    while (bin_reader_ptr_->Read(&key, &val)) {
      /*
      if (key == "SINGA_VERSION") {
        CHECK(version_ == std::stoi(val)) << key << " in .bin and .desc
      mismatch: "
          << val << " (bin) vs " << version_ << " (desc)";
        continue;
      }
      */

      CHECK(param_names_.count(key) == 0);
      param_names_.insert(key);
      CHECK(tp.ParseFromString(val));
      param_map_[key].FromProto(tp);
    }
    // need ro set version_ by getting data form param_map_["SINGA_VERSION"]?
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
  //  bin_writer_ptr_->Flush();

  std::string desc_str = "parameter name: " + key;
  Shape shape = param.shape();
  desc_str += "\tdata type: " + std::to_string(param.data_type());
  desc_str += "\tdim: " + std::to_string(shape.size());
  desc_str += "\tshape:";
  for (size_t s : shape) desc_str += " " + std::to_string(s);
  text_writer_ptr_->Write(key, desc_str);
  // text_writer_ptr_->Flush();
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
