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

#include "singa/worker.h"
#include "singa/neuralnet/layer.h"
#include "singa/neuralnet/input_layer.h"
#include "singa/neuralnet/neuron_layer.h"
#include "singa/neuralnet/loss_layer.h"

#include <cblas.h>
#include <glog/logging.h>
#include <math.h>
#include <cfloat>
#include "singa/utils/factory.h"
#include "singa/utils/singleton.h"
#include "singa/utils/math_blob.h"

namespace singa {

using std::string;

void Layer::SetupLayer(Layer* layer, const string str, const vector<Layer*>& srclayers) {
  LayerProto layer_conf;
  layer_conf.ParseFromString(str);
  layer->Setup(layer_conf, srclayers);
  for (auto param : layer->GetParams())
      param->InitValues();
}

Layer* Layer::CreateLayer(const string str) {
  LayerProto layer_conf;
  layer_conf.ParseFromString(str);
  return Layer::Create(layer_conf);
}

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
  string ret = "";
  if ((flag & kForward) == kForward && data_.count() !=0) {
    ret += StringPrintf("data:%e ", Asum(data_));
    for (Param* p : GetParams())
      ret += StringPrintf("%s:%13.9f ",
          p->name().c_str(), Asum(p->data()));
  }
  if ((flag & kBackward) == kBackward && grad_.count() != 0) {
    ret += StringPrintf("grad:%e ", Asum(grad_));
    for (Param* p : GetParams())
      ret += StringPrintf("%s:%13.9f ",
          p->name().c_str(), Asum(p->grad()));
  }
  return ret;
}
}  // namespace singa
