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

#include "singa/model/optimizer.h"
#include "singa/core/device.h"
#include "singa/core/tensor.h"
#include "singa/utils/logging.h"

#include <memory>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <string>
#include <unordered_map>

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
  int total_num_;
  std::shared_ptr<Device> dev_;
  std::unordered_map<std::string, int> aggr_count_, copy_count_;
  std::unordered_map<std::string, Tensor> grad_buffer_,
    param_buffer_, partial_sum_;
  std::unordered_map<std::string, bool> has_init_;
  std::unordered_map<std::string, std::mutex> mtx_;
  std::unordered_map<std::string, std::condition_variable>
    aggr_count_eq_total_num_;
};
}  //  namespace singa

#endif  //  SINGA_MODEL_UPDATER_H_
