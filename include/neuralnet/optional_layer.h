#ifndef SINGA_NEURALNET_OPTIONAL_LAYER_H_
#define SINGA_NEURALNET_OPTIONAL_LAYER_H_

#ifdef USE_LMDB
#include <lmdb.h>
#endif
#include <string>
#include "neuralnet/base_layer.h"

namespace singa {

#ifdef USE_LMDB
class LMDBDataLayer : public DataLayer {
 public:
  ~LMDBDataLayer();

  void Setup(const LayerProto& proto, int npartitions) override;
  void OpenLMDB(const std::string& path);
  void ComputeFeature(Phase phase, Metric *perf) override;
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

}  // namespace singa

#endif  // SINGA_NEURALNET_OPTIONAL_LAYER_H_
