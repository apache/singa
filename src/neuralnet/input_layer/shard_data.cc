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

#include "singa/neuralnet/input_layer/shard_data.h"
#include "mshadow/tensor.h"
#include "singa/utils/image_transform.h"
#include "singa/utils/tokenizer.h"
namespace singa {

using namespace mshadow;
using mshadow::cpu;
using mshadow::Shape4;
using mshadow::Tensor;

using std::string;
using std::vector;

ShardDataLayer::~ShardDataLayer() {
  if (shard_ != nullptr)
    delete shard_;
  shard_ = nullptr;
}

void ShardDataLayer::Setup(const LayerProto& proto,
    const vector<Layer*>& srclayers) {
  Layer::Setup(proto, srclayers);
  shard_ = new DataShard(proto.sharddata_conf().path(), DataShard::kRead);
  string key;
  shard_->Next(&key, &sample_);
  delete shard_;
  shard_ = nullptr;
  batchsize_ = proto.sharddata_conf().batchsize();
  if (partition_dim() == 0)
    batchsize_ /= proto.num_partitions();
  records_.resize(batchsize_);
  random_skip_ = proto.sharddata_conf().random_skip();
}

void ShardDataLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  if (shard_ == nullptr)
    shard_ = new DataShard(layer_conf_.sharddata_conf().path(),
                           DataShard::kRead);
  if (random_skip_) {
    int nskip = rand() % random_skip_;
    LOG(INFO) << "Random Skip " << nskip << " records, there are "
              << shard_->Count() << " records in total";
    string key;
    for (int i = 0; i < nskip; i++) {
      shard_->Next(&key, &sample_);
    }
    random_skip_ = 0;
  }
  for (auto& record : records_) {
    string key;
    if (!shard_->Next(&key, &record)) {
      shard_->SeekToFirst();
      CHECK(shard_->Next(&key, &record));
    }
  }
}

} // namespace singa
