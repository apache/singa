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

#include "singa/neuralnet/input_layer/lmdb_data.h"
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

#ifdef USE_LMDB
LMDBDataLayer::~LMDBDataLayer() {
  mdb_cursor_close(mdb_cursor_);
  mdb_txn_abort(mdb_txn_);
  mdb_cursor_ = nullptr;
}

void LMDBDataLayer::Setup(const LayerProto& proto,
    const vector<Layer*>& srclayers) {
  Layer::Setup(proto, srclayers);
  OpenLMDB(proto.lmdbdata_conf().path());
  CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT),
           MDB_SUCCESS);
  mdb_cursor_close(mdb_cursor_);
  mdb_txn_abort(mdb_txn_);
  mdb_cursor_ = nullptr;
  CaffeDatum datum;
  datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
  SingleLabelImageRecord* record = sample_.mutable_image();
  ConvertCaffeDatumToRecord(datum, record);
  batchsize_ = proto.lmdbdata_conf().batchsize();
  if (partition_dim() == 0)
    batchsize_ /= proto.num_partitions();
  records_.resize(batchsize_);
  random_skip_ = proto.lmdbdata_conf().random_skip();
}

void LMDBDataLayer::OpenLMDB(const std::string& path) {
  CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
  CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS);  // 1TB
  CHECK_EQ(mdb_env_open(mdb_env_, path.c_str(),
           MDB_RDONLY, 0664), MDB_SUCCESS) << "cannot open lmdb " << path;
  CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
      << "mdb_txn_begin failed";
  CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
      << "mdb_open failed";
  CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
      << "mdb_cursor_open failed";
  LOG(INFO) << "Opening lmdb " << path;
  CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
           MDB_SUCCESS) << "mdb_cursor_get failed";
}

void LMDBDataLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  if (mdb_cursor_ == nullptr)
    OpenLMDB(layer_conf_.lmdbdata_conf().path());
  if (random_skip_) {
    int nskip = rand() % random_skip_;
    int n = 0;
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
             &mdb_value_, MDB_FIRST), MDB_SUCCESS);
    while (mdb_cursor_get(mdb_cursor_, &mdb_key_,
           &mdb_value_, MDB_NEXT) == MDB_SUCCESS)
      n++;
    LOG(INFO) << "Random Skip " << nskip << " records of total "
              << n << "records";
    // We have reached the end. Restart from the first.
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
             &mdb_value_, MDB_FIRST), MDB_SUCCESS);
    for (int i = 0; i < nskip; i++) {
      if (mdb_cursor_get(mdb_cursor_, &mdb_key_,
          &mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
                 &mdb_value_, MDB_FIRST), MDB_SUCCESS);
      }
    }
    random_skip_ = 0;
  }
  CaffeDatum datum;
  for (auto& record : records_) {
    SingleLabelImageRecord* image = record.mutable_image();
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
             &mdb_value_, MDB_GET_CURRENT), MDB_SUCCESS);
    datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
    ConvertCaffeDatumToRecord(datum, image);
    if (mdb_cursor_get(mdb_cursor_, &mdb_key_,
        &mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
               &mdb_value_, MDB_FIRST), MDB_SUCCESS);
    }
  }
}

void LMDBDataLayer::ConvertCaffeDatumToRecord(const CaffeDatum& datum,
                                              SingleLabelImageRecord* record) {
  record->set_label(datum.label());
  record->clear_shape();
  if (datum.has_channels())
    record->add_shape(datum.channels());
  if (datum.has_height())
    record->add_shape(datum.height());
  if (datum.has_width())
    record->add_shape(datum.width());
  if (datum.has_data())
    record->set_pixel(datum.data());
  if (datum.float_data_size()) {
    record->clear_data();
    for (float x : datum.float_data())
      record->add_data(x);
  }
}
#endif

} // namespace singa
