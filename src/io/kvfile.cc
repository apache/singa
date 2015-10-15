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

#include "singa/io/kvfile.h"

#include <glog/logging.h>

namespace singa {
namespace io {

KVFile::KVFile(const std::string& path, Mode mode, int capacity) :
path_(path), mode_(mode), capacity_(capacity) {
  buf_ = new char[capacity];
  switch (mode) {
    case KVFile::kRead:
      fdat_.open(path_, std::ios::in | std::ios::binary);
      if (!fdat_.is_open()) {
        // path may be a directory
        path_ = path + "/shard.dat";
        fdat_.open(path_, std::ios::in | std::ios::binary);
      }
      CHECK(fdat_.is_open()) << "Cannot create file " << path_;
      break;
    case KVFile::kCreate:
      fdat_.open(path_, std::ios::binary | std::ios::out | std::ios::trunc);
      CHECK(fdat_.is_open()) << "Cannot create file " << path_;
      break;
    case KVFile::kAppend:
      fdat_.open(path_, std::ios::in | std::ios::binary);
      if (!fdat_.is_open()) {
        // path may be a directory
        path_ = path + "/shard.dat";
        fdat_.open(path_, std::ios::in | std::ios::binary);
      }
      CHECK(fdat_.is_open()) << "Cannot open file " << path_;
      fdat_.close();
      {
        int last_tuple = PrepareForAppend(path_);
        fdat_.open(path_, std::ios::binary | std::ios::out
            | std::ios::in | std::ios::ate);
        fdat_.seekp(last_tuple);
      }
      break;
    default:
      LOG(FATAL) << "unknown model to open KVFile " << mode;
      break;
  }
}

KVFile::~KVFile() {
  if (mode_ != kRead)
    Flush();
  delete[] buf_;
  fdat_.close();
}
#ifdef USE_PROTOBUF
bool KVFile::Next(std::string* key, google::protobuf::Message* val) {
  int vallen = Next(key);
  if (vallen == 0) return false;
  val->ParseFromArray(buf_ + offset_, vallen);
  offset_ += vallen;
  return true;
}

bool KVFile::Insert(const std::string& key,
    const google::protobuf::Message& val) {
  std::string str;
  val.SerializeToString(&str);
  return Insert(key, str);
}
#endif

bool KVFile::Next(std::string *key, std::string* val) {
  int vallen = Next(key);
  if (vallen == 0) return false;
  val->clear();
  for (int i = 0; i < vallen; ++i)
    val->push_back(buf_[offset_ + i]);
  offset_ += vallen;
  return true;
}

// insert one complete tuple
bool KVFile::Insert(const std::string& key, const std::string& val) {
  if (keys_.find(key) != keys_.end() || val.size() == 0)
    return false;
  int size = key.size() + val.size() + 2*sizeof(size_t);
  if (bufsize_ + size > capacity_) {
    fdat_.write(buf_, bufsize_);
    bufsize_ = 0;
    CHECK_LE(size, capacity_) << "Tuple size is larger than capacity "
      << "Try a larger capacity size";
  }
  *reinterpret_cast<size_t*>(buf_ + bufsize_) = key.size();
  bufsize_ += sizeof(size_t);
  memcpy(buf_ + bufsize_, key.data(), key.size());
  bufsize_ += key.size();
  *reinterpret_cast<size_t*>(buf_ + bufsize_) = val.size();
  bufsize_ += sizeof(size_t);
  memcpy(buf_ + bufsize_, val.data(), val.size());
  bufsize_ += val.size();
  return true;
}

void KVFile::SeekToFirst() {
  CHECK_EQ(mode_, kRead);
  bufsize_ = 0;
  offset_ = 0;
  fdat_.clear();
  fdat_.seekg(0);
  CHECK(fdat_.is_open()) << "Cannot create file " << path_;
}

void KVFile::Flush() {
  fdat_.write(buf_, bufsize_);
  fdat_.flush();
  bufsize_ = 0;
}

int KVFile::Count() {
  std::ifstream fin(path_, std::ios::in | std::ios::binary);
  CHECK(fdat_.is_open()) << "Cannot create file " << path_;
  int count = 0;
  while (true) {
    size_t len;
    fin.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (!fin.good()) break;
    fin.seekg(len, std::ios_base::cur);
    if (!fin.good()) break;
    fin.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (!fin.good()) break;
    fin.seekg(len, std::ios_base::cur);
    if (!fin.good()) break;
    count++;
  }
  fin.close();
  return count;
}

int KVFile::Next(std::string *key) {
  key->clear();
  int ssize = sizeof(size_t);
  if (!PrepareNextField(ssize)) return 0;
  int keylen = *reinterpret_cast<size_t*>(buf_ + offset_);
  offset_ += ssize;
  if (!PrepareNextField(keylen)) return 0;
  for (int i = 0; i < keylen; ++i)
    key->push_back(buf_[offset_ + i]);
  offset_ += keylen;
  if (!PrepareNextField(ssize)) return 0;
  int vallen = *reinterpret_cast<size_t*>(buf_ + offset_);
  offset_ += ssize;
  if (!PrepareNextField(vallen)) return 0;
  return vallen;
}

int KVFile::PrepareForAppend(const std::string& path) {
  std::ifstream fin(path, std::ios::in | std::ios::binary);
  if (!fin.is_open()) return 0;
  int last_tuple_offset = 0;
  char buf[256];
  size_t len;
  while (true) {
    fin.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (!fin.good()) break;
    fin.read(buf, len);
    buf[len] = '\0';
    if (!fin.good()) break;
    fin.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (!fin.good()) break;
    fin.seekg(len, std::ios_base::cur);
    if (!fin.good()) break;
    keys_.insert(std::string(buf));
    last_tuple_offset = fin.tellg();
  }
  fin.close();
  return last_tuple_offset;
}

// if the buf does not have the next complete field, read data from disk
bool KVFile::PrepareNextField(int size) {
  if (offset_ + size > bufsize_) {
    bufsize_ -= offset_;
    // wangsh: commented, not sure what this check does
    // CHECK_LE(bufsize_, offset_);
    for (int i = 0; i < bufsize_; ++i)
      buf_[i] = buf_[i + offset_];
    offset_ = 0;
    if (fdat_.eof()) {
      return false;
    } else {
      fdat_.read(buf_ + bufsize_, capacity_ - bufsize_);
      bufsize_ += fdat_.gcount();
      if (size > bufsize_) return false;
    }
  }
  return true;
}

}  // namespace io
}  // namespace singa
