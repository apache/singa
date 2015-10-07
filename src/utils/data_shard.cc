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

#include "singa/utils/data_shard.h"

#include <glog/logging.h>
#include <sys/stat.h>

namespace singa {

DataShard::DataShard(const std::string& folder, int mode)
    : DataShard(folder, mode , 104857600) {}

DataShard::DataShard(const std::string& folder, int mode, int capacity) {
  struct stat sb;
  if (stat(folder.c_str(), &sb) == 0 && S_ISDIR(sb.st_mode)) {
    LOG(INFO) << "Open shard folder " << folder;
  } else {
    LOG(FATAL) << "Cannot open shard folder " << folder;
  }
  path_ = folder + "/shard.dat";
  switch (mode) {
    case DataShard::kRead: {
      fdat_.open(path_, std::ios::in | std::ios::binary);
      CHECK(fdat_.is_open()) << "Cannot create file " << path_;
      break;
    }
    case DataShard::kCreate: {
      fdat_.open(path_, std::ios::binary | std::ios::out | std::ios::trunc);
      CHECK(fdat_.is_open()) << "Cannot create file " << path_;
      break;
    }
    case DataShard::kAppend: {
      int last_tuple = PrepareForAppend(path_);
      fdat_.open(path_, std::ios::binary | std::ios::out | std::ios::in
                 | std::ios::ate);
      CHECK(fdat_.is_open()) << "Cannot create file " << path_;
      fdat_.seekp(last_tuple);
      break;
    }
  }
  mode_ = mode;
  offset_ = 0;
  bufsize_ = 0;
  capacity_ = capacity;
  buf_ = new char[capacity];
}

DataShard::~DataShard() {
  delete buf_;
  fdat_.close();
}

bool DataShard::Next(std::string* key, google::protobuf::Message* val) {
  int vallen = Next(key);
  if (vallen == 0) return false;
  val->ParseFromArray(buf_ + offset_, vallen);
  offset_ += vallen;
  return true;
}

bool DataShard::Next(std::string *key, std::string* val) {
  int vallen = Next(key);
  if (vallen == 0) return false;
  val->clear();
  for (int i = 0; i < vallen; ++i)
    val->push_back(buf_[offset_ + i]);
  offset_ += vallen;
  return true;
}

bool DataShard::Insert(const std::string& key,
                       const google::protobuf::Message& val) {
  std::string str;
  val.SerializeToString(&str);
  return Insert(key, str);
}

// insert one complete tuple
bool DataShard::Insert(const std::string& key, const std::string& val) {
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

void DataShard::SeekToFirst() {
  CHECK_EQ(mode_, kRead);
  bufsize_ = 0;
  offset_ = 0;
  fdat_.close();
  fdat_.open(path_, std::ios::in | std::ios::binary);
  CHECK(fdat_.is_open()) << "Cannot create file " << path_;
}

void DataShard::Flush() {
  fdat_.write(buf_, bufsize_);
  fdat_.flush();
  bufsize_ = 0;
}

int DataShard::Count() {
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

int DataShard::Next(std::string *key) {
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

int DataShard::PrepareForAppend(const std::string& path) {
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
bool DataShard::PrepareNextField(int size) {
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

}  // namespace singa
