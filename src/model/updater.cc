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

#include "singa/model/updater.h"

namespace singa {

void Updater::Setup(const OptimizerConf& conf) { opt_->Setup(conf); }
void Updater::Register(const string& name, const ParamSpec& specs) {
  opt_->Register(name, specs);
  aggr_count_[name] = 0;
  copy_count_[name] = 0;
}
void Updater::Apply(int step, const string& name, Tensor& grad, Tensor& value) {
  CHECK(partial_count_.count(name) == 1) << "Parameter " << name
                                         << " has not been registered before.";
  std::unique_lock<std::mutex> aggr_lock(aggr_mtx_);
  if (partial_count_[name] == 0) {
    partial_sum_[name].ResetLike(grad);
    partial_sum_[name].ToHost();
    buffer_[name].ResetLike(partial_sum_[name]);
  }
  buffer_[name].CopyData(grad);
  Add(partial_sum_[name], buffer_[name], &partial_sum_[name]);
  ++partial_count_[name];

  // Now we get enought paritial gradient from all neural net instances,
  // then we calcuate the average gradient.
  if (partial_count_[name] == total_num_) {
    Div(partial_sum_[name], static_cast<float>(total_num_),
        &partial_sum_[name]);
    buffer_[name].CopyData(value);
    // Apply optimization algorithm based on the aggregated gradient.
    opt_->Apply(step, name, partial_sum_[name], buffer_[name]);
    copy_count_[name] = 0;
    partial_count_eq_total_num_.notify_all();
  } else {
    // Block this thread when we have not gotten enough paritial gradients.
    while (partial_count_[name] != total_num_) {
      partial_count_eq_total_num_.wait(lock);
    }
  }
  lock.unlock();
  value.CopyData(buffer_[name]);
  lock.lock();
  ++copy_count_[name];
  if (copy_count_[name] == total_num_) partial_count_[name] = 0;
}
}  // namesapce singa
