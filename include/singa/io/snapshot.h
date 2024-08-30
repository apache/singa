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

#ifndef SINGA_UTILS_SNAPSHOT_H_
#define SINGA_UTILS_SNAPSHOT_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "singa/core/tensor.h"
#include "singa/io/reader.h"
#include "singa/io/writer.h"
#include "singa/proto/core.pb.h"
#include "singa/utils/logging.h"

namespace singa {
/// The snapshot management.
/// It dumps the model parameter snapshot as checkpoint files, which coud be
/// used for fine-tuning and deployment.
/// The model paramters are separated from model definition, i.e., net
/// construction. Users either randomly initialize the layer parameters or using
/// the parameters from checkpoint files using Snapshot after creating the
/// neural network.
class Snapshot {
 public:
  enum Mode { kRead, kWrite };
  /// <prefix>.model is the binary file for parameter key-value pair.
  /// <prefix>.meta is the text file describing information about paramters,
  /// i.e.
  /// name and shape, one line per parameter.
  /// kRead for reading snapshot, whereas kWrite for dumping out snapshot.
  /// max_param_size: in MB
  Snapshot(const std::string& prefix, Mode mode, int max_param_size = 10);
  ~Snapshot() {}
  /// Read parameters saved as tensors from checkpoint file.
  std::vector<std::pair<std::string, Tensor>> Read();
  /// Read parameter shapes from description file.
  std::vector<std::pair<std::string, Shape>> ReadShape();
  /// Read parameter returned as a tensor for a given parameter name.
  Tensor Read(const std::string& Key);
  /// Read parameter shape for a given parameter name.
  Shape ReadShape(const std::string& key);
  /// Serialize and dump out parameter. This method will write two files, one
  /// binary file is for serialized tensors, the other csv file is for parameter
  /// names and shapes.
  void Write(const std::string& key, const Tensor& param);
  /// available for singa > 1.0.1
  int version() const { return version_; }

 private:
  /// version of SINGA which generates the snapshot
  int version_ = 0;
  std::string prefix_;
  Mode mode_;
  std::unique_ptr<io::BinFileWriter> bin_writer_ptr_;
  std::unique_ptr<io::Writer> text_writer_ptr_;
  std::unique_ptr<io::BinFileReader> bin_reader_ptr_;
  /// Check whether parameter name is unique.
  std::unordered_set<std::string> param_names_;
  /// Preload key-parameter tensor pairs for seeking a specified key.
  std::unordered_map<std::string, Tensor> param_map_;
};
}  //  namespace singa

#endif  //  SINGA_UTILS_SNAPSHOT_H_
