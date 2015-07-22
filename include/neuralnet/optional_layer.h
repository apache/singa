#ifdef USE_OPTIONAL_LAYER
#ifndef SINGA_NEURALNET_OPTIONAL_LAYER_
#define SINGA_NEURALNET_OPTIONAL_LAYER_
#include "neuralnet/layer.h"

namespace singa {

class LMDBDataLayer: public DataLayer{
 public:
  using Layer::ComputeFeature;

  void Setup(const LayerProto& proto, int npartitions) override;
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
} /* singa */

#endif  // SINGA_NEURALNET_OPTIONAL_LAYER_
#endif

