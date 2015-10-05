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

#ifndef SINGA_NEURALNET_INPUT_LAYER_H_
#define SINGA_NEURALNET_INPUT_LAYER_H_

#include <string>
#include <vector>
#include "neuralnet/layer.h"
#include "utils/data_shard.h"
#include "io/store.h"
/**
 * \file this file includes the declarations of input layers that inherit the
 * base InputLayer to load input features.
 *
 * The feature loading phase can be implemented using a single layer or
 * separated into DataLayer (for loading features as records) and ParserLayer
 * (for parsing features from records). SINGA has provided some subclasses of
 * DataLayer and ParserLayer.
 *
 * Data prefetching can be implemented as a sub-class of InputLayer.
 * SINGA provides a built-in PrefetchLayer which embeds DataLayer and
 * ParserLayer.
 */
namespace singa {
using std::string;
using std::vector;

/************************Start of new input layers***************************/
/**
 * Base class for loading data from Store.
 */
class StoreInputLayer : virtual public InputLayer {
 public:
  ~StoreInputLayer();
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;

  ConnectionType dst_layer_connection() const override { return kOneToMany; }

 protected:
  /**
   * Parsing the (key, val) tuple to get feature (and label).
   * Subclasses must implment this function.
   * @param[in] k parse this tuple as the k-th instance of one mini-batch.
   * @param[in] flag used to guide the parsing, e.g., kDeploy phase should not
   * parse labels from the tuple.
   * @param[in] key
   * @param[in] val
   */
  virtual bool Parse(int k, int flag, const string& key, const string& val) = 0;

 protected:
  int batchsize_;
  io::Store* store_ = nullptr;
};

/**
 * Base layer for parsing a key-value tuple as a feature vector with fixed
 * length. The feature shape is indicated by users in the configuration.
 * Each tuple may has a label.
 */
class SingleLabelRecordLayer : public StoreInputLayer {
 public:
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;

 protected:
  /**
   * Load a single record (tuple), e.g., the mean or standard variance vector.
   */
  virtual void LoadRecord(const string& backend, const string& path,
      Blob<float>* to) = 0;

 protected:
  /**
   * Feature standardization by processing each feature dimension via
   * @f$ y = (x - mu)/ std @f$
   * <a href= "http://ufldl.stanford.edu/wiki/index.php/Data_Preprocessing">
   * UFLDL</a>
   */
  Blob<float> mean_, std_;
};

/**
 * Specific layer that parses the value string loaded by Store into a
 * SingleLabelImageRecord.
 */
class ProtoRecordLayer : public SingleLabelRecordLayer {
 public:
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;

 protected:
  /**
   * Parse key as instance ID and val into SingleLabelImageRecord.
   * @copydetails StoreInputLayer::Parse()
   */
  bool Parse(int k, int flag, const string& key, const string& val) override;
  void LoadRecord(const string& backend,
                  const string& path,
                  Blob<float>* to) override;

 private:
  // TODO(wangwei) decode the image
  bool encoded_;
};

/**
 * Specific layer that parses the value string loaded by Store as a line from
 * a CSV file.
 *
 * It assumes the first column is the label except that has_label_ is configured
 * to false. Or the data is used in deploy mode.
 */
class CSVRecordLayer : public SingleLabelRecordLayer {
 public:
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;

 protected:
  bool Parse(int k, int flag, const string& key, const string& val) override;
  void LoadRecord(const string& backend,
                  const string& path,
                  Blob<float>* to) override;

 private:
  std::string sep_;
  bool has_label_;
};

/**
 * Do preprocessing for images, including cropping, mirroring, resizing.
 */
class ImagePreprocessLayer : public InputLayer {
 public:
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers);

 private:
  bool mirror_ = false;
  int cropsize_ = 0;
  int resize_ = 0;
  float scale_ = 1;
};

/************************End of new input layers***************************/
/**
 * Base layer for reading ::Record  from local Shard, HDFS, lmdb, etc.
 */
class DataLayer: virtual public InputLayer {
 public:
  Blob<float>* mutable_data(const Layer* layer) override { return nullptr; }
  ConnectionType dst_layer_connection() const override {
    return kOneToMany;
  }

  inline int batchsize() const { return batchsize_; }
  virtual const Record& sample() const {
    return sample_;
  }
  /**
   * @return the loaded records
   */
  virtual const std::vector<Record>& records() const {
    return records_;
  }

 protected:
  int random_skip_;
  int batchsize_;
  Record sample_;
  std::vector<Record> records_;
};
/**
 * Layer for loading Record from DataShard.
 *
 * It is derived from DataLayer.
 */
class ShardDataLayer : public DataLayer {
 public:
  ~ShardDataLayer();

  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;

 private:
  DataShard* shard_;
};

#ifdef USE_LMDB
#include <lmdb.h>
class LMDBDataLayer : public DataLayer {
 public:
  ~LMDBDataLayer();

  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void OpenLMDB(const std::string& path);
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ConvertCaffeDatumToRecord(const CaffeDatum& datum,
                                 SingleLabelImageRecord* record);

 private:
  MDB_env* mdb_env_;
  MDB_dbi mdb_dbi_;
  MDB_txn* mdb_txn_;
  MDB_cursor* mdb_cursor_;
  MDB_val mdb_key_, mdb_value_;
};
#endif

/**
 * Base layer for parsing the input records into Blobs.
 */
class ParserLayer : public InputLayer {
 public:
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override {}
  ConnectionType dst_layer_connection() const override {
    return kOneToMany;
  }
  /**
   * Parse records from DataLayer into blob.
   */
  virtual void ParseRecords(int flag, const std::vector<Record>& records,
      Blob<float>* blob) = 0;
};

/**
 * Derived from ParserLayer to parse label from SingaleLabelImageRecord.
 */
class LabelLayer : public ParserLayer {
 public:
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ParseRecords(int flag, const std::vector<Record>& records,
                    Blob<float>* blob) override;
};

/**
 * Derived from ParserLayer to parse MNIST feature from SingaleLabelImageRecord.
 */
class MnistLayer : public ParserLayer {
 public:
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ParseRecords(int flag, const std::vector<Record>& records,
                    Blob<float>* blob) override;

 protected:
  float norm_a_, norm_b_;
};
/**
 * Derived from ParserLayer to parse RGB image feature from
 * SingaleLabelImageRecord.
 */
class RGBImageLayer : public ParserLayer {
 public:
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ParseRecords(int flag, const std::vector<Record>& records,
                    Blob<float>* blob) override;

 private:
  float scale_;
  int cropsize_;
  bool mirror_;
  Blob<float> mean_;
};
/**
 * Layer for prefetching data records and parsing them.
 *
 * The data loading and parsing work is done by internal DataLayer and
 * ParserLayer respectively. This layer controls the prefetching thread, i.e.,
 * creating and joining the prefetching thread.
 */
class PrefetchLayer : public Layer {
 public:
  ~PrefetchLayer();
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;
  void ComputeGradient(int flag, const vector<Layer*>& srclayers) override {}

 protected:
  std::thread thread_;
};

}  // namespace singa

#endif  // SINGA_NEURALNET_INPUT_LAYER_H_
