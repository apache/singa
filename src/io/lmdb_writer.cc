/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef DISABLE_WARNINGS

#include "singa/io/writer.h"
#include "singa/utils/logging.h"

#ifdef USE_LMDB

namespace singa {
namespace io {
bool LMDBWriter::Open(const std::string& path, Mode mode) {
  path_ = path;
  mode_ = mode;
  MDB_CHECK(mdb_env_create(&mdb_env_));
  if (mode_ != kCreate && mode_ != kAppend) {
    LOG(FATAL) << "unknown mode to open LMDB" << mode_;
    return false;
  }
  if (mode_ == kCreate)
    // It will fail if there is a dir at "path"
    CHECK_EQ(mkdir(path.c_str(), 0744), 0) << "mkdir " << path << " failed";
  int flags = 0;
  int rc = mdb_env_open(mdb_env_, path.c_str(), flags, 0664);
#ifndef ALLOW_LMDB_NOLOCK
  MDB_CHECK(rc);
#else
  if (rc == EACCES) {
    LOG(WARNING) << "Permission denied. Trying with MDB_NOLOCK ...";
    // Close and re-open environment handle
    mdb_env_close(mdb_env_);
    MDB_CHECK(mdb_env_create(&mdb_env_));
    // Try again with MDB_NOLOCK
    flags |= MDB_NOLOCK;
    MDB_CHECK(mdb_env_open(mdb_env_, path.c_str(), flags, 0664));
  } else
    MDB_CHECK(rc);
#endif
  return true;
}

void LMDBWriter::Close() {
  Flush();
  if (mdb_env_ != nullptr) {
    mdb_env_close(mdb_env_);
    mdb_env_ = nullptr;
  }
}

bool LMDBWriter::Write(const std::string& key, const std::string& value) {
  CHECK_NE(key, "") << "Key is an empty string!";
  keys.push_back(key);
  values.push_back(value);
  return true;
}

// Flush is to "commit to DB"
void LMDBWriter::Flush() {
  if (keys.size() == 0) return;
  MDB_dbi mdb_dbi;
  MDB_val mdb_key, mdb_data;
  MDB_txn* mdb_txn;

  // Initialize MDB variables
  MDB_CHECK(mdb_txn_begin(mdb_env_, NULL, 0, &mdb_txn));
  MDB_CHECK(mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi));

  for (size_t i = 0; i < keys.size(); i++) {
    mdb_key.mv_size = keys[i].size();
    mdb_key.mv_data = const_cast<char*>(keys[i].data());
    mdb_data.mv_size = values[i].size();
    mdb_data.mv_data = const_cast<char*>(values[i].data());

    // Add data to the transaction
    int put_rc = mdb_put(mdb_txn, mdb_dbi, &mdb_key, &mdb_data, 0);
    CHECK_NE(put_rc, MDB_KEYEXIST) << "Key already exist: " << keys[i];
    if (put_rc == MDB_MAP_FULL) {
      // Out of memory - double the map size and retry
      mdb_txn_abort(mdb_txn);
      mdb_dbi_close(mdb_env_, mdb_dbi);
      DoubleMapSize();
      Flush();
      return;
    }
    // May have failed for some other reason
    MDB_CHECK(put_rc);
  }

  // Commit the transaction
  int commit_rc = mdb_txn_commit(mdb_txn);
  if (commit_rc == MDB_MAP_FULL) {
    // Out of memory - double the map size and retry
    mdb_dbi_close(mdb_env_, mdb_dbi);
    DoubleMapSize();
    Flush();
    return;
  }
  // May have failed for some other reason
  MDB_CHECK(commit_rc);

  // Cleanup after successful commit
  mdb_dbi_close(mdb_env_, mdb_dbi);
  keys.clear();
  values.clear();
}

void LMDBWriter::DoubleMapSize() {
  struct MDB_envinfo current_info;
  MDB_CHECK(mdb_env_info(mdb_env_, &current_info));
  size_t new_size = current_info.me_mapsize * 2;
  LOG(INFO) << "Doubling LMDB map size to " << (new_size >> 20) << "MB ...";
  MDB_CHECK(mdb_env_set_mapsize(mdb_env_, new_size));
}

inline void LMDBWriter::MDB_CHECK(int mdb_status) {
  CHECK_EQ(mdb_status, MDB_SUCCESS) << mdb_strerror(mdb_status);
}
}  // namespace io
}  // namespace singa
#endif  // USE_LMDB

#endif
