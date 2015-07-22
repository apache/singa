#ifdef USE_OPTIONAL_LAYER
#include "neuralnet/optional_layer.h"
namespace singa {

/*********************LMDBDataLayer**********************************/
void LMDBDataLayer::ComputeFeature(Phase phase, Metric* perf){
  if(random_skip_){
    int nskip = rand() % random_skip_;
    int n=0;
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
          &mdb_value_, MDB_FIRST), MDB_SUCCESS);
    while (mdb_cursor_get(mdb_cursor_, &mdb_key_,
          &mdb_value_, MDB_NEXT) == MDB_SUCCESS)
      n++;
    LOG(INFO)<<"Random Skip "<<nskip<<" records of total "<<n<<"records";
    // We have reached the end. Restart from the first.
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
          &mdb_value_, MDB_FIRST), MDB_SUCCESS);
    for(int i=0;i<nskip;i++){
      if (mdb_cursor_get(mdb_cursor_, &mdb_key_,
            &mdb_value_, MDB_NEXT) != MDB_SUCCESS) {
        // We have reached the end. Restart from the first.
        DLOG(INFO) << "Restarting data prefetching from start.";
        CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_,
              &mdb_value_, MDB_FIRST), MDB_SUCCESS);
      }
    }
    random_skip_=0;
  }
  CaffeDatum datum;
  for(auto& record: records_){
    SingleLabelImageRecord* image=record.mutable_image();
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
    SingleLabelImageRecord* record){
  record->set_label(datum.label());
  record->clear_shape();
  if(datum.has_channels())
    record->add_shape(datum.channels());
  if(datum.has_height())
    record->add_shape(datum.height());
  if(datum.has_width())
    record->add_shape(datum.width());
  if(datum.has_data())
    record->set_pixel(datum.data());
  if(datum.float_data_size()){
    record->clear_data();
    for(float x: datum.float_data())
      record->add_data(x);
  }
}

void LMDBDataLayer::Setup(const LayerProto& proto, int npartitions) {
  Layer::Setup(proto, npartitions);
  CHECK_EQ(mdb_env_create(&mdb_env_), MDB_SUCCESS) << "mdb_env_create failed";
  CHECK_EQ(mdb_env_set_mapsize(mdb_env_, 1099511627776), MDB_SUCCESS); // 1TB
  CHECK_EQ(mdb_env_open(mdb_env_,
        proto.lmdbdata_conf().path().c_str(),
        MDB_RDONLY, 0664), MDB_SUCCESS) << "cannot open lmdb "
    << proto.lmdbdata_conf().path();
  CHECK_EQ(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_), MDB_SUCCESS)
    << "mdb_txn_begin failed";
  CHECK_EQ(mdb_open(mdb_txn_, NULL, 0, &mdb_dbi_), MDB_SUCCESS)
    << "mdb_open failed";
  CHECK_EQ(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_), MDB_SUCCESS)
    << "mdb_cursor_open failed";
  LOG(INFO) << "Opening lmdb " << proto.lmdbdata_conf().path();
  CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_FIRST),
      MDB_SUCCESS) << "mdb_cursor_get failed";

  if (mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, MDB_NEXT)
      != MDB_SUCCESS) {
    CHECK_EQ(mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_,
          MDB_FIRST), MDB_SUCCESS);
  }
  CaffeDatum datum;
  datum.ParseFromArray(mdb_value_.mv_data, mdb_value_.mv_size);
  SingleLabelImageRecord* record=sample_.mutable_image();
  ConvertCaffeDatumToRecord(datum, record);

  batchsize_=batchsize();
  if(partition_dim() == 0)
    batchsize_ /= npartitions;
  records_.resize(batchsize_);
  random_skip_=proto.lmdbdata_conf().random_skip();
}

} /* singa */

#endif

