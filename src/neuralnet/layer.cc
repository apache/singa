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

#include "singa/neuralnet/layer.h"

#include <cblas.h>
#include <glog/logging.h>
#include <math.h>
#include <cfloat>
#include "singa/utils/factory.h"
#include "singa/utils/singleton.h"
#include "singa/utils/math_blob.h"

namespace singa {

using std::string;

Layer* Layer::Create(const LayerProto& proto) {
  auto* factory = Singleton<Factory<Layer>>::Instance();
  Layer* layer = nullptr;
  if (proto.has_user_type())
    layer = factory->Create(proto.user_type());
  else
    layer = factory->Create(proto.type());
  return layer;
}

const std::string Layer::ToString(bool debug, int flag) {
  if (!debug)
    return "";
  string ret = StringPrintf("Layer %10s ", name().c_str());
  if ((flag & kForward) == kForward && data_.count() !=0) {
    ret += StringPrintf("data norm1 %13.9f", Asum(cpu, data_));
  } else if ((flag & kBackward) == kBackward) {
    if (grad_.count() != 0)
      ret += StringPrintf("grad norm1 %13.9f\n", Asum(cpu, grad_));
  }
  if ((flag & kTrain) == kTrain) {
    for (Param* p : GetParams()) {
      ret += StringPrintf(
          "param id %2d, name %10s, value norm1 %13.9f, grad norm1 %13.9f\n",
          p->id(), p->name().c_str(), Asum(cpu, p->data()),
          Asum(cpu, p->grad()));
    }
  }
  return ret;
}

const std::string LossLayer::ToString(bool debug, int flag) {
  std::string disp;
  if (debug) {
    disp = Layer::ToString(debug, flag);
  } else {
    disp = metric_.ToLogString();
    metric_.Reset();
  }
  return disp;
}
}  // namespace singa
