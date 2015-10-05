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

#include "neuralnet/input_layer.h"
#include "mshadow/tensor.h"
#include "utils/image_transform.h"
#include "utils/tokenizer.h"
namespace singa {

using namespace mshadow;
using mshadow::cpu;
using mshadow::Shape4;
using mshadow::Tensor;

using std::string;
using std::vector;

/*****************ImagePreprocess**************************************/
void ImagePreprocessLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  CHECK_EQ(srclayers.size(), 1);
  InputLayer::Setup(conf, srclayers);
  scale_ = conf.rgbimage_conf().scale();
  cropsize_ = conf.rgbimage_conf().cropsize();
  mirror_ = conf.rgbimage_conf().mirror();
  const auto& src = srclayers.at(0)->data(this);
  const auto& shape = src.shape();
  CHECK_EQ(shape.size(), 4);
  CHECK_EQ(shape.at(2), shape.at(3));
  if (cropsize_ != 0 && cropsize_ != shape.at(2)) {
    data_.Reshape(vector<int>{shape.at(0), shape.at(1), cropsize_, cropsize_});
  } else {
    data_ = src;
  }
}

void ImagePreprocessLayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  const auto& srcdata = srclayers.at(0)->data(this);
  int batchsize = srcdata.shape()[0], channel = srcdata.shape()[1];
  int height = srcdata.shape()[2], width = srcdata.shape()[3];
  const float* srcdptr = srcdata.cpu_data();
  float* dptr = data_.mutable_cpu_data();
  int srcimage_size = channel * height * width;
  int image_size = channel * data_.shape()[2] * data_.shape()[3];
  for (int k = 0; k < batchsize; k++) {
    int h_offset = 0, w_offset = 0;
    if (cropsize_> 0 && ((flag & kTrain) == kTrain)) {
      h_offset = rand() % (srcdata.shape()[1] - cropsize_);
      w_offset = rand() % (srcdata.shape()[2] - cropsize_);
    }
    bool do_mirror = mirror_ && rand() % 2 && ((flag & kTrain) == kTrain);
    ImageTransform(srcdptr + k * srcimage_size, nullptr, do_mirror, cropsize_,
        cropsize_, h_offset, w_offset, srcdata.shape()[1], height, width,
        scale_, dptr + image_size);
  }
}

/*************StoreInputLayer******************/
StoreInputLayer::~StoreInputLayer() {
  if (store_ != nullptr) {
    delete store_;
  }
}

void StoreInputLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  InputLayer::Setup(conf, srclayers);
  batchsize_ = conf.store_conf().batchsize();
}

void StoreInputLayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  string key, val;
  if (store_ == nullptr) {
    store_ = io::OpenStore(layer_conf_.store_conf().backend(),
                             layer_conf_.store_conf().path(),
                             io::kRead);
  }
  for (int k = 0; k < batchsize_; k++){
    if (!store_->Read(&key, &val)) {
      store_->SeekToFirst();
      CHECK(store_->Read(&key, &val));
    }
    // TODO(wangwei) random skip and shuffle among this mini-batch
    Parse(k, flag, key, val);
  }
}
/*********SingleLabelRecordLayer******************/
void SingleLabelRecordLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  StoreInputLayer::Setup(conf, srclayers);

  vector<int> shape {batchsize_};
  for (int s : conf.store_conf().shape())
    shape.push_back(s);
  data_.Reshape(shape);
  aux_data_.resize(batchsize_);
}
void SingleLabelRecordLayer::ComputeFeature(int flag,
    const vector<Layer*>& srclayers) {
  StoreInputLayer::ComputeFeature(flag, srclayers);

  auto& store_conf = layer_conf_.store_conf();
  if (store_conf.has_mean_file() && mean_.count() == 0) {
    mean_.Reshape(vector<int>{data_.count() / batchsize_});
    LoadRecord(store_conf.backend(), store_conf.mean_file(), &mean_);
  } else if (store_conf.has_mean_value() && mean_.count() == 0) {
    mean_.Reshape(vector<int>{data_.count() / batchsize_});
    for (int i = 0; i < data_.count() / batchsize_; i++)
      mean_.mutable_cpu_data()[i] = store_conf.mean_value();
  }
  if (store_conf.has_std_file() && std_.count() == 0) {
    std_.Reshape(vector<int>{data_.count() / batchsize_});
    LoadRecord(store_conf.backend(), store_conf.std_file(), &std_);
    // TODO(wangwei) check std[i] != 0
  } else if (store_conf.has_std_value() && std_.count() == 0) {
    std_.Reshape(vector<int>{data_.count() / batchsize_});
    CHECK_NE(store_conf.std_value(), 0);
    for (int i = 0; i < data_.count() / batchsize_; i++)
      std_.mutable_cpu_data()[i] = store_conf.std_value();
  }

  if (mean_.count()) {
    const float* mean = mean_.cpu_data();
    for (int k = 0; k < batchsize_; k++){
      float* dptr = data_.mutable_cpu_data() + k * mean_.count();
      for (int i = 0; i < mean_.count(); i++) {
        dptr[i] -= mean[i];
      }
    }
  }
  if (std_.count()) {
    const float* std = std_.cpu_data();
    for (int k = 0; k < batchsize_; k++){
      float* dptr = data_.mutable_cpu_data() + k * std_.count();
      for (int i = 0; i < std_.count(); i++) {
        dptr[i] /= std[i];
      }
    }
  }
}
/*****************CSVRecordLayer*******************/
void CSVRecordLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  SingleLabelRecordLayer::Setup(conf, srclayers);
  sep_ = conf.store_conf().separator();
}

void CSVRecordLayer::LoadRecord(const string& backend,
    const string&path, Blob<float>* to) {
  io::Store* store = io::OpenStore(backend, path, io::kRead);
  string key, val;
  CHECK(store->Read(&key, &val));
  float* ptr = to->mutable_cpu_data();
  Tokenizer t(val, sep_);
  string x;
  for (int i = 0; i< to->count(); i++) {
    t >> x;
    ptr[i] = stof(x);
  }
  CHECK(!t.Valid());
  delete store;
}

bool CSVRecordLayer::Parse(int k, int flag, const string& key,
    const string& value) {
  float* ptr = data_.mutable_cpu_data() + k * data_.count() / batchsize_;
  Tokenizer t(value, sep_);
  string x;
  // parse label if not deploy phase and has_label is set.
  if ((flag & kDeploy) == 0 && layer_conf_.store_conf().has_label()) {
    t >> x;
    aux_data_[k] = stoi(x);
  }
  for (int i = 0; i< data_.count() / batchsize_; i++) {
    t >> x;
    ptr[i] = stof(x);
  }
  CHECK(!t.Valid());
  return true;
}


/*********ProtoRecordLayer******************/
void ProtoRecordLayer::Setup(const LayerProto& conf,
    const vector<Layer*>& srclayers) {
  SingleLabelRecordLayer::Setup(conf, srclayers);
  encoded_ = conf.store_conf().encoded();
}

void ProtoRecordLayer::LoadRecord(const string& backend,
    const string&path, Blob<float>* to) {
  io::Store* store = io::OpenStore(backend, path, io::kRead);
  string key, val;
  CHECK(store->Read(&key, &val));
  SingleLabelImageRecord image;
  image.ParseFromString(val);
  CHECK_EQ(to->count(), image.data_size());
  float* ptr = to->mutable_cpu_data();
  for (int i = 0; i< to->count(); i++)
    ptr[i] = image.data(i);
  delete store;
}

bool ProtoRecordLayer::Parse(int k, int flag, const string& key,
    const string& value) {
  SingleLabelImageRecord image;
  image.ParseFromString(value);
  int size = data_.count() / batchsize_;
  if (image.data_size()) {
    CHECK_EQ(size, image.data_size());
    float* ptr = data_.mutable_cpu_data() + k * size;
    for (int i = 0; i< size; i++)
      ptr[i] = image.data(i);
  } else if (image.pixel().size()) {
    CHECK_EQ(size, image.pixel().size());
    float* ptr = data_.mutable_cpu_data() + k * size;
    string pixel = image.pixel();
    for (int i = 0; i < size; i++)
      ptr[i] =  static_cast<float>(static_cast<uint8_t>(pixel[i]));
  } else {
    LOG(ERROR) << "not pixel nor pixel";
  }
  if ((flag & kDeploy) == 0) {  // deploy mode does not have label
    aux_data_.at(k) = image.label();
  }
  return true;
}

/************* Implementation for ParserLayer ***********/
void ParserLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  CHECK_EQ(srclayers.size(), 1);
  auto datalayer = dynamic_cast<DataLayer*>(*srclayers.begin());
  ParseRecords(flag, datalayer->records(), &data_);
}

#ifdef USE_LMDB
/*********************LMDBDataLayer**********************************/
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


/***************Implementation for ShardDataLayer**************************/
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

/********* Implementation for LabelLayer **************/
void LabelLayer::Setup(const LayerProto& proto,
    const vector<Layer*>& srclayers) {
  Layer::Setup(proto, srclayers);
  CHECK_EQ(srclayers.size(), 1);
  int batchsize = dynamic_cast<DataLayer*>(srclayers[0])->batchsize();
  data_.Reshape(vector<int>{batchsize});
}

void LabelLayer::ParseRecords(int flag, const vector<Record>& records,
    Blob<float>* blob) {
  int rid = 0;
  float *label = blob->mutable_cpu_data();
  for (const Record& record : records) {
    label[rid++] = record.image().label();
    // CHECK_LT(record.image().label(),10);
  }
  CHECK_EQ(rid, blob->shape()[0]);
}
/**************** Implementation for MnistLayer ******************/
void MnistLayer::ParseRecords(int flag, const vector<Record>& records,
    Blob<float>* blob) {
  LOG_IF(ERROR, records.size() == 0) << "Empty records to parse";
  int ndim = records.at(0).image().shape_size();
  int inputsize = records.at(0).image().shape(ndim-1);
  CHECK_EQ(inputsize, blob->shape()[2]);

  float* dptr = blob->mutable_cpu_data();
  for (const Record& record : records) {
    const SingleLabelImageRecord& imagerecord = record.image();
    if (imagerecord.pixel().size()) {
      string pixel = imagerecord.pixel();
      for (int i = 0, k = 0; i < inputsize; i++) {
        for (int j = 0; j < inputsize; j++) {
          // NOTE!!! must cast pixel to uint8_t then to float!!! waste a lot of
          // time to debug this
          float x =  static_cast<float>(static_cast<uint8_t>(pixel[k++]));
          x = x / norm_a_-norm_b_;
          *dptr = x;
          dptr++;
        }
      }
    } else {
      for (int i = 0, k = 0; i < inputsize; i++) {
        for (int j = 0; j < inputsize; j++) {
          *dptr = imagerecord.data(k++) / norm_a_ - norm_b_;
          dptr++;
        }
      }
    }
  }
  CHECK_EQ(dptr, blob->mutable_cpu_data() + blob->count());
}

void MnistLayer::Setup(const LayerProto& proto,
    const vector<Layer*>& srclayers) {
  Layer::Setup(proto, srclayers);
  CHECK_EQ(srclayers.size(), 1);
  int batchsize = dynamic_cast<DataLayer*>(srclayers[0])->batchsize();
  Record sample = dynamic_cast<DataLayer*>(srclayers[0])->sample();
  norm_a_ = proto.mnist_conf().norm_a();
  norm_b_ = proto.mnist_conf().norm_b();
  int ndim = sample.image().shape_size();
  CHECK_GE(ndim, 2);
  int s = sample.image().shape(ndim - 1);
  CHECK_EQ(s, sample.image().shape(ndim - 2));
  data_.Reshape(vector<int>{batchsize, 1, s, s});
}

/*************** Implementation for RGBImageLayer *************************/
void RGBImageLayer::ParseRecords(int flag, const vector<Record>& records,
    Blob<float>* blob) {
  const vector<int>& s = blob->shape();
  Tensor<cpu, 4> images(data_.mutable_cpu_data(),
      Shape4(s[0], s[1], s[2], s[3]));
  const SingleLabelImageRecord& r = records.at(0).image();
  Tensor<cpu, 3> raw_image(Shape3(r.shape(0), r.shape(1), r.shape(2)));
  AllocSpace(raw_image);
  Tensor<cpu, 3> croped_image(nullptr, Shape3(s[1], s[2], s[3]));
  if (cropsize_)
    AllocSpace(croped_image);
  int rid = 0;
  const float* meandptr = mean_.cpu_data();
  for (const Record& record : records) {
    auto image = images[rid];
    bool do_crop = cropsize_> 0 && ((flag & kTrain) == kTrain);
    bool do_mirror = mirror_ && rand() % 2 && ((flag & kTrain) == kTrain);
    float* dptr = nullptr;
    if (do_crop || do_mirror)
      dptr = raw_image.dptr;
    else
      dptr = image.dptr;
    if (record.image().pixel().size()) {
      string pixel = record.image().pixel();
      for (size_t i = 0; i < pixel.size(); i++)
        dptr[i] = static_cast<float>(static_cast<uint8_t>(pixel[i]));
    } else {
      memcpy(dptr, record.image().data().data(),
          sizeof(float) * record.image().data_size());
    }
    for (int i = 0; i < mean_.count(); i++)
      dptr[i] -= meandptr[i];
    if (do_crop) {
      int hoff = rand() % (r.shape(1) - cropsize_);
      int woff = rand() % (r.shape(2) - cropsize_);
      Shape<2> cropshape = Shape2(cropsize_, cropsize_);
      if (do_mirror) {
        croped_image = expr::crop(raw_image, cropshape, hoff, woff);
        image = expr::mirror(croped_image);
      } else {
        image = expr::crop(raw_image, cropshape, hoff, woff);
      }
    } else if (do_mirror) {
      image = expr::mirror(raw_image);
    }
    rid++;
  }
  if (scale_)
    images = images * scale_;
  FreeSpace(raw_image);
  if (cropsize_)
    FreeSpace(croped_image);
}

void RGBImageLayer::Setup(const LayerProto& proto,
    const vector<Layer*>& srclayers) {
  ParserLayer::Setup(proto, srclayers);
  CHECK_EQ(srclayers.size(), 1);
  scale_ = proto.rgbimage_conf().scale();
  cropsize_ = proto.rgbimage_conf().cropsize();
  mirror_ = proto.rgbimage_conf().mirror();
  int batchsize = dynamic_cast<DataLayer*>(srclayers[0])->batchsize();
  Record sample = dynamic_cast<DataLayer*>(srclayers[0])->sample();
  vector<int> shape;
  shape.push_back(batchsize);
  for (int x : sample.image().shape()) {
    shape.push_back(x);
  }
  CHECK_EQ(shape.size(), 4);
  if (cropsize_) {
    shape[2] = cropsize_;
    shape[3] = cropsize_;
  }
  data_.Reshape(shape);
  mean_.Reshape({shape[1], shape[2], shape[3]});
  if (proto.rgbimage_conf().has_meanfile()) {
    if (proto.rgbimage_conf().meanfile().find("binaryproto") != string::npos) {
      CaffeBlob mean;
      ReadProtoFromBinaryFile(proto.rgbimage_conf().meanfile().c_str(), &mean);
      CHECK_EQ(mean_.count(), mean.data_size());
      memcpy(mean_.mutable_cpu_data(), mean.data().data(),
             sizeof(float)*mean.data_size());
    } else {
      SingleLabelImageRecord mean;
      ReadProtoFromBinaryFile(proto.rgbimage_conf().meanfile().c_str(), &mean);
      CHECK_EQ(mean_.count(), mean.data_size());
      memcpy(mean_.mutable_cpu_data(), mean.data().data(),
             sizeof(float)*mean.data_size());
    }
  } else {
    memset(mean_.mutable_cpu_data(), 0, sizeof(float) * mean_.count());
  }
}

/************* Implementation for PrefetchLayer ***********/
PrefetchLayer::~PrefetchLayer() {
  if (thread_.joinable())
    thread_.join();
}


void PrefetchLayer::ComputeFeature(int flag, const vector<Layer*>& srclayers) {
  LOG(FATAL) << "Not implemented";
}

}  // namespace singa
