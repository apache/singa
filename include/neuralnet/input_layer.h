#ifndef SINGA_NEURALNET_INPUT_LAYER_H_
#define SINGA_NEURALNET_INPUT_LAYER_H_

#include <vector>
#include "neuralnet/layer.h"
#include "utils/data_shard.h"
/**
 * \file this file includes the declarations of input layers that inherit the
 * base InputLayer to load input features.
 *
 * The feature loading phase can be implemented using a single layer or
 * separated into DataLayer (for loading features as records) and ParserLayer
 * (for parsing features from records). SINGA has provided some built-in layers
 * for DataLayer and ParserLayer.
 *
 * Data prefetching can be implemented as a sub-class of InputLayer.
 * SINGA provides a built-in PrefetchLayer which embeds DataLayer and
 * ParserLayer.
 */
namespace singa {
/**
 * Base layer for reading records from local Shard, HDFS, lmdb, etc.
 */
class DataLayer: public InputLayer {
 public:
  void ComputeGradient(int flag, Metric* perf) override {}
  Blob<float>* mutable_data(const Layer* layer) override {
    return nullptr;
  }
  Blob<float>* mutable_grad(const Layer* layer) override {
    return nullptr;
  }
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

  void Setup(const LayerProto& proto, int npartitions) override;
  void ComputeFeature(int flag, Metric *perf) override;

 private:
  DataShard* shard_;
};

#ifdef USE_LMDB
#include <lmdb.h>
class LMDBDataLayer : public DataLayer {
 public:
  ~LMDBDataLayer();

  void Setup(const LayerProto& proto, int npartitions) override;
  void OpenLMDB(const std::string& path);
  void ComputeFeature(int flag, Metric *perf) override;
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
  void ComputeFeature(int flag, Metric* perf) override;
  void ComputeGradient(int flag, Metric* perf) override {}
  ConnectionType dst_layer_connection() const override {
    return kOneToMany;
  }
  /**
   * Parse records from DataLayer into blob.
   */
  virtual void ParseRecords(int flag, const std::vector<Record>& records,
      Blob<float>* blob) = 0;
  Blob<float>* mutable_grad(const Layer* layer) override {
    return nullptr;
  }
  const Blob<float>& grad(const Layer* from) const  override {
    CHECK(false) << "Parser layer has not gradient blob";
    return grad_;
  }
};

/**
 * Derived from ParserLayer to parse label from SingaleLabelImageRecord.
 */
class LabelLayer : public ParserLayer {
 public:
  void Setup(const LayerProto& proto, int npartitions) override;
  void ParseRecords(int flag, const std::vector<Record>& records,
                    Blob<float>* blob) override;
};

/**
 * Derived from ParserLayer to parse MNIST feature from SingaleLabelImageRecord.
 */
class MnistLayer : public ParserLayer {
 public:
  void Setup(const LayerProto& proto, int npartitions) override;
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
  void Setup(const LayerProto& proto, int npartitions) override;
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
  void ComputeFeature(int flag, Metric* perf) override;
  void ComputeGradient(int flag, Metric* perf) override {}

 protected:
  std::thread thread_;
};
}  // namespace singa

#endif  // SINGA_NEURALNET_INPUT_LAYER_H_
