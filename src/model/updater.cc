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
  has_averaged_[name] = false;
  has_init_[name] = false;
}
void Updater::Apply(int step, const string& name, Tensor& grad, Tensor& value) {
  CHECK(aggr_count_.count(name) == 1) << "Parameter " << name
                                         << " has not been registered before.";
  /// This lock is aimed to protect aggregation counter, data transfering buffer,
  /// and partial aggregation result. However, the data transfering can be moved out
  /// of the critial section to improve performance in the future.
  std::unique_lock<std::mutex> lock(mtx_);
  if (aggr_count_[name] == 0) {
    if (!has_init_[name]) {
      partial_sum_[name].ResetLike(grad);
      partial_sum_[name].ToHost();
      buffer_[name].ResetLike(partial_sum_[name]);
      has_init_[name] = true;
    } else {
      partial_sum_[name].SetValue(.0f);
    }
  }
  buffer_[name].CopyData(grad);
  Add(partial_sum_[name], buffer_[name], &partial_sum_[name]);
  ++aggr_count_[name];

  /// Block this thread when we have not gotten enough paritial gradients.
  if (aggr_count_[name] != total_num_) {
    while (aggr_count_[name] != total_num_) {
      aggr_count_eq_total_num_.wait(lock);
    }
  } else {
    aggr_count_eq_total_num_.notify_all();
  }

  /// Now we get enought paritial gradient from all neural net instances,
  /// then we calcuate the average gradient. The first notified thread
  /// finish the averaging once.
  if (!has_averaged_[name]) {
    Div(partial_sum_[name], static_cast<float>(total_num_),
        &partial_sum_[name]);
    copy_count_[name] = 0;
    has_averaged_[name] = true;
  }

  buffer_[name].CopyData(value);
  /// Apply optimization algorithm based on the averaged gradient.
  opt_->Apply(step, name, partial_sum_[name], buffer_[name]);
  value.CopyData(buffer_[name]);

  /// The last thread finishing copy should set aggregation counter back to 0.
  ++copy_count_[name];
  if (copy_count_[name] == total_num_) {
    aggr_count_[name] = 0;
    has_averaged_[name] = false;
  }
}
}  // namesapce singa
