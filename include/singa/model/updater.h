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

#ifndef SINGA_MODEL_UPDATER_H_
#define SINGA_MODEL_UPDATER_H_

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "singa/core/device.h"
#include "singa/core/tensor.h"
#include "singa/model/optimizer.h"
#include "singa/utils/logging.h"

namespace singa {
/// Basic Updater class just forward all the method function call
/// to the wrapped Optimizer.
class Updater {
 public:
  explicit Updater(Optimizer* opt) : opt_{opt} {}
  virtual ~Updater() {}
  /// Forward Setup() to Optimizer.
  virtual void Setup(const OptimizerConf& conf);
  /// Forward Register() to Optimizer.
  virtual void Register(const string& name, const ParamSpec& specs);
  /// Forward Apply() to Optimizer.
  virtual void Apply(int step, const string& name, Tensor& grad, Tensor& value);
  Optimizer* GetOptimizer() { return opt_; }

  // No copy allowed.
  Updater(const Updater&) = delete;
  void operator=(const Updater&) = delete;

 protected:
  Optimizer* opt_;
};

/// LocalUpdater do gradient aggregation and update gradient calling
/// the wrapped Optimizer on a specific device (i.e., CPU or GPU).
class LocalUpdater : public Updater {
 public:
  LocalUpdater(int total_num, Optimizer* opt,
               std::shared_ptr<Device> dev = defaultDevice)
      : Updater(opt), total_num_{total_num}, dev_(dev) {}
  virtual ~LocalUpdater() override {}
  /// Forward Register() to Optimizer.
  virtual void Register(const string& name, const ParamSpec& specs) override;
  /// Update parameter value based on given gradient by invoking optimizer
  /// algoritim. When tranining net call this function will be blocked until
  /// all the partial gradients are aggrageted in a synchronized style training.
  virtual void Apply(int step, const string& name, Tensor& grad,
                     Tensor& value) override;

 private:
  template <typename T1, typename T2>
  struct key_hasher {
    size_t operator()(const std::pair<T1, T2>& p) const {
      auto h1 = std::hash<T1>{}(p.first);
      auto h2 = std::hash<T2>{}(p.second);
      return h1 ^ h2;
    }
  };

  int total_num_;
  std::shared_ptr<Device> dev_;
  std::unordered_map<std::string, std::atomic<int>> dev_index_;
  std::unordered_map<std::string, int> to_updater_finished_;
  std::unordered_map<std::pair<int, std::string>, Tensor,
                     key_hasher<int, std::string>>
      grad_buffer_;
  std::unordered_map<std::string, Tensor> sum_, param_buffer_;
  std::unordered_map<std::string, std::mutex> mtx_;
  std::unordered_map<std::string, std::condition_variable>
      to_updater_all_finished_;
};
}  //  namespace singa

#endif  //  SINGA_MODEL_UPDATER_H_
