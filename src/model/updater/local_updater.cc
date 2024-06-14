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

#include <vector>

#include "singa/model/updater.h"

namespace singa {

void LocalUpdater::Register(const string& name, const ParamSpec& specs) {
  opt_->Register(name, specs);
  param_buffer_[name];
  param_buffer_[name].ToDevice(dev_);
  sum_[name];
  sum_[name].ToDevice(dev_);
  for (int i = 0; i < total_num_; ++i) {
    grad_buffer_[std::make_pair(i, name)];
    grad_buffer_[std::make_pair(i, name)].ToDevice(dev_);
  }
  dev_index_[name] = 0;
  to_updater_finished_[name] = 0;
  mtx_[name];
}

void LocalUpdater::Apply(int step, const string& name, Tensor& grad,
                         Tensor& value) {
  CHECK(param_buffer_.count(name) == 1)
      << "Parameter " << name << " has not been registered before.";
  int nth = dev_index_[name]++;
  auto key = std::make_pair(nth, name);
  if (grad_buffer_[key].Size() != grad.Size()) {
    grad_buffer_[key].Resize(grad.shape());
    grad_buffer_[key].AsType(grad.data_type());
  }
  grad_buffer_[key].CopyData(grad);

  std::unique_lock<std::mutex> lock(mtx_[name]);
  ++to_updater_finished_[name];
  if (to_updater_finished_[name] != total_num_) {
    while (to_updater_finished_[name] > 0) {
      to_updater_all_finished_[name].wait(lock);
    }
  } else {
    if (param_buffer_[name].Size() != value.Size()) {
      param_buffer_[name].Resize(value.shape());
      param_buffer_[name].AsType(value.data_type());
      param_buffer_[name].CopyData(value);
      sum_[name].ResetLike(param_buffer_[name]);
    }
    sum_[name].SetValue(.0f);
    for (int i = 0; i < total_num_; ++i)
      Add(sum_[name], grad_buffer_[std::make_pair(i, name)], &sum_[name]);
    Div(sum_[name], static_cast<float>(total_num_), &sum_[name]);
    opt_->Apply(step, name, sum_[name], param_buffer_[name]);
    to_updater_finished_[name] = 0;
    dev_index_[name] = 0;
    to_updater_all_finished_[name].notify_all();
  }
  lock.unlock();
  value.CopyData(param_buffer_[name]);
}

}  // namespace singa
