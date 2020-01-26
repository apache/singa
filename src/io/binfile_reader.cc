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

#include "singa/io/reader.h"
#include "singa/utils/logging.h"

namespace singa {
namespace io {
bool BinFileReader::Open(const std::string& path) {
  path_ = path;
  return OpenFile();
}

bool BinFileReader::Open(const std::string& path, int capacity) {
  path_ = path;
  capacity_ = capacity;
  return OpenFile();
}

void BinFileReader::Close() {
  if (buf_ != nullptr) {
    delete[] buf_;
    buf_ = nullptr;
  }
  if (fdat_.is_open()) fdat_.close();
}

bool BinFileReader::Read(std::string* key, std::string* value) {
  CHECK(fdat_.is_open()) << "File not open!";
  char magic[4];
  int smagic = sizeof(magic);
  if (!PrepareNextField(smagic)) return false;
  memcpy(magic, buf_ + offset_, smagic);
  offset_ += smagic;

  if (magic[0] == kMagicWord[0] && magic[1] == kMagicWord[1]) {
    if (magic[2] != 0 && magic[2] != 1)
      LOG(FATAL) << "File format error: magic word does not match!";
    if (magic[2] == 1)
      if (!ReadField(key)) return false;
    if (!ReadField(value)) return false;
  } else {
    LOG(FATAL) << "File format error: magic word does not match!";
  }
  return true;
}

int BinFileReader::Count() {
  std::ifstream fin(path_, std::ios::in | std::ios::binary);
  CHECK(fin.is_open()) << "Cannot create file " << path_;
  int count = 0;
  while (true) {
    size_t len;
    char magic[4];
    fin.read(reinterpret_cast<char*>(magic), sizeof(magic));
    if (!fin.good()) break;
    if (magic[2] == 1) {
      fin.read(reinterpret_cast<char*>(&len), sizeof(len));
      if (!fin.good()) break;
      fin.seekg(len, std::ios_base::cur);
      if (!fin.good()) break;
    }
    fin.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (!fin.good()) break;
    fin.seekg(len, std::ios_base::cur);
    if (!fin.good()) break;
    count++;
  }
  fin.close();
  return count;
}

void BinFileReader::SeekToFirst() {
  bufsize_ = 0;
  offset_ = 0;
  fdat_.clear();
  fdat_.seekg(0);
  CHECK(fdat_.is_open()) << "Cannot create file " << path_;
}

bool BinFileReader::OpenFile() {
  buf_ = new char[capacity_];
  fdat_.open(path_, std::ios::in | std::ios::binary);
  if (!fdat_.is_open()) LOG(WARNING) << "Cannot open file " << path_;
  return fdat_.is_open();
}

bool BinFileReader::ReadField(std::string* content) {
  content->clear();
  int ssize = sizeof(size_t);
  if (!PrepareNextField(ssize)) return false;
  int len = *reinterpret_cast<int*>(buf_ + offset_);
  offset_ += ssize;
  if (!PrepareNextField(len)) return false;
  content->reserve(len);
  content->insert(0, buf_ + offset_, len);
  // for (int i = 0; i < len; ++i) content->push_back(buf_[offset_ + i]);
  offset_ += len;
  return true;
}

// if the buf does not have the next complete field, read data from disk
bool BinFileReader::PrepareNextField(int size) {
  if (offset_ + size > bufsize_) {
    bufsize_ -= offset_;
    memcpy(buf_, buf_ + offset_, bufsize_);
    offset_ = 0;
    if (fdat_.eof()) {
      return false;
    } else {
      fdat_.read(buf_ + bufsize_, capacity_ - bufsize_);
      bufsize_ += (int)fdat_.gcount();
      CHECK_LE(size, bufsize_) << "Field size is too large: " << size;
    }
  }
  return true;
}

}  // namespace io
}  // namespace singa
