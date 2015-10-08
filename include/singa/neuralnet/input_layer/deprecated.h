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

#ifndef SINGA_NEURALNET_INPUT_LAYER_DEPRECATED_H_
#define SINGA_NEURALNET_INPUT_LAYER_DEPRECATED_H_

#include "singa/neuralnet/layer.h"
#include "singa/io/kvfile.h"
namespace singa {
/**
 * @deprecated please use the StoreInputLayer.
 *
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
 * @deprecated Please use the subclasses of StoreInputLayer.
 *
 * Layer for loading Record from DataShard.
 */
class ShardDataLayer : public DataLayer {
 public:
  ~ShardDataLayer();

  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ComputeFeature(int flag, const vector<Layer*>& srclayers) override;

 private:
  DataShard* shard_;
};
/**
 * @deprecated please use the subclasses of StoreInputLayer.
 *
 * Layer for loading Record from LMDB.
 */
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

/******************Parser layers***************/
/**
 * @deprecated Please use the subclasses of StoreInputLayer which load and parse
 * data in a single layer.
 *
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
 *
 * @deprecated Please use the SingleLabelRecordLayer which parses both feature
 * and label for each record. Its aux_data() function returns the parsed labels.
 *
 * Derived from ParserLayer to parse label in SingaleLabelImageRecord loaded by
 * ShardDataLayer.
 */
class LabelLayer : public ParserLayer {
 public:
  void Setup(const LayerProto& proto, const vector<Layer*>& srclayers) override;
  void ParseRecords(int flag, const std::vector<Record>& records,
                    Blob<float>* blob) override;
};

/**
 * @deprecated Please use the subclasses of StoreInputLayer.
 *
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
 * @deprecated please use the ImagePreprocessLayer which preprocess image
 * feature from data Blob of source layers.
 *
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
}  // namespace singa

#endif  // SINGA_NEURALNET_INPUT_LAYER_DEPRECATED_H_
