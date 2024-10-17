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

#include "singa/io/reader.h"
#include "singa/utils/logging.h"

#ifdef USE_LMDB

namespace singa {
namespace io {
bool LMDBReader::Open(const std::string& path) {
  path_ = path;
  MDB_CHECK(mdb_env_create(&mdb_env_));
  int flags = MDB_RDONLY | MDB_NOTLS;
  int rc = mdb_env_open(mdb_env_, path_.c_str(), flags, 0664);
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
    MDB_CHECK(mdb_env_open(mdb_env_, source.c_str(), flags, 0664));
  } else {
    MDB_CHECK(rc);
  }
#endif
  MDB_CHECK(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_));
  MDB_CHECK(mdb_dbi_open(mdb_txn_, NULL, 0, &mdb_dbi_));
  MDB_CHECK(mdb_cursor_open(mdb_txn_, mdb_dbi_, &mdb_cursor_));
  SeekToFirst();
  return true;
}

void LMDBReader::Close() {
  if (mdb_env_ != nullptr) {
    mdb_cursor_close(mdb_cursor_);
    mdb_txn_abort(mdb_txn_);
    mdb_dbi_close(mdb_env_, mdb_dbi_);
    mdb_env_close(mdb_env_);
    mdb_env_ = nullptr;
    mdb_txn_ = nullptr;
    mdb_cursor_ = nullptr;
  }
}

bool LMDBReader::Read(std::string* key, std::string* value) {
  if (first_ != true) Seek(MDB_NEXT);
  if (valid_ == false) return false;
  *key = string(static_cast<const char*>(mdb_key_.mv_data), mdb_key_.mv_size);
  *value =
      string(static_cast<const char*>(mdb_value_.mv_data), mdb_value_.mv_size);
  first_ = false;
  return true;
}

int LMDBReader::Count() {
  MDB_env* env;
  MDB_dbi dbi;
  MDB_txn* txn;
  MDB_cursor* cursor;
  int flags = MDB_RDONLY | MDB_NOTLS | MDB_NOLOCK;
  MDB_CHECK(mdb_env_create(&env));
  MDB_CHECK(mdb_env_open(env, path_.c_str(), flags, 0664));
  MDB_CHECK(mdb_txn_begin(env, NULL, MDB_RDONLY, &txn));
  MDB_CHECK(mdb_dbi_open(txn, NULL, 0, &dbi));
  MDB_CHECK(mdb_cursor_open(txn, dbi, &cursor));
  int status = MDB_SUCCESS;
  int count = 0;
  MDB_val key, value;
  while (true) {
    status = mdb_cursor_get(cursor, &key, &value, MDB_NEXT);
    if (status == MDB_NOTFOUND) break;
    count++;
  }
  mdb_cursor_close(cursor);
  mdb_txn_abort(txn);
  mdb_dbi_close(env, dbi);
  mdb_env_close(env);
  return count;
}

void LMDBReader::SeekToFirst() {
  Seek(MDB_FIRST);
  first_ = true;
}

void LMDBReader::Seek(MDB_cursor_op op) {
  int mdb_status = mdb_cursor_get(mdb_cursor_, &mdb_key_, &mdb_value_, op);
  if (mdb_status == MDB_NOTFOUND) {
    valid_ = false;
  } else {
    MDB_CHECK(mdb_status);
    valid_ = true;
  }
}

inline void LMDBReader::MDB_CHECK(int mdb_status) {
  CHECK_EQ(mdb_status, MDB_SUCCESS) << mdb_strerror(mdb_status);
}
}  // namespace io
}  // namespace singa
#endif  // USE_LMDB

#endif
