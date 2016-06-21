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

#include "singa/io/writer.h"
#include "singa/utils/logging.h"

namespace singa {
namespace io {
bool BinFileWriter::Open(const std::string& path, Mode mode, int capacity) {
  CHECK(!fdat_.is_open());
  path_ = path;
  mode_ = mode;
  capacity_ = capacity;
  buf_ = new char[capacity_];
  switch (mode) {
    case kCreate:
      fdat_.open(path_, std::ios::binary | std::ios::out | std::ios::trunc);
      CHECK(fdat_.is_open()) << "Cannot create file " << path_;
      break;
    case kAppend:
      fdat_.open(path_, std::ios::in | std::ios::binary);
      CHECK(fdat_.is_open()) << "Cannot open file " << path_;
      fdat_.close();
      {
        int last_tuple = PrepareForAppend(path_);
        fdat_.open(path_, std::ios::binary | std::ios::out | std::ios::in |
                              std::ios::ate);
        fdat_.seekp(last_tuple);
      }
      break;
    default:
      LOG(FATAL) << "unknown model to open KVFile " << mode;
      break;
  }
  return fdat_.is_open();
}

void BinFileWriter::Close() {
  Flush();
  if (buf_ != nullptr) {
    delete buf_;
    buf_ = nullptr;
  }
  if (fdat_.is_open()) fdat_.close();
}

bool BinFileWriter::Write(const std::string& key, const std::string& value) {
  CHECK(fdat_.is_open()) << "File not open!";
  if (value.size() == 0) return false;
  // magic_word + (key_len + key) + val_len + val
  char magic[4];
  int size;
  memcpy(magic, kMagicWord, sizeof(kMagicWord));
  magic[3] = 0;
  if (key.size() == 0) {
    magic[2] = 0;
    size = sizeof(magic) + sizeof(size_t) + value.size();
  } else {
    magic[2] = 1;
    size = sizeof(magic) + 2 * sizeof(size_t) + key.size() + value.size();
  }

  if (bufsize_ + size > capacity_) {
    fdat_.write(buf_, bufsize_);
    bufsize_ = 0;
    CHECK_LE(size, capacity_) << "Tuple size is larger than capacity "
                              << "Try a larger capacity size";
  }

  memcpy(buf_ + bufsize_, magic, sizeof(magic));
  bufsize_ += sizeof(magic);
  if (key.size() > 0) {
    *reinterpret_cast<size_t*>(buf_ + bufsize_) = key.size();
    bufsize_ += sizeof(size_t);
    std::memcpy(buf_ + bufsize_, key.data(), key.size());
    bufsize_ += key.size();
  }
  *reinterpret_cast<size_t*>(buf_ + bufsize_) = value.size();
  bufsize_ += sizeof(size_t);
  std::memcpy(buf_ + bufsize_, value.data(), value.size());
  bufsize_ += value.size();
  return true;
}

void BinFileWriter::Flush() {
  CHECK(fdat_);
  if (bufsize_ > 0) {
    fdat_.write(buf_, bufsize_);
    fdat_.flush();
    bufsize_ = 0;
  }
}

int BinFileWriter::PrepareForAppend(const std::string& path) {
  std::ifstream fin(path, std::ios::in | std::ios::binary);
  if (!fin.is_open()) return 0;
  int last_tuple_offset = 0;
  char buf[256];
  size_t len;
  char magic[4];
  while (true) {
    fin.read(magic, sizeof(magic));
    if (!fin.good()) break;
    if (magic[2] == 1) {
      fin.read(reinterpret_cast<char*>(&len), sizeof(len));
      if (!fin.good()) break;
      fin.read(buf, len);
      buf[len] = '\0';
      if (!fin.good()) break;
    }
    fin.read(reinterpret_cast<char*>(&len), sizeof(len));
    if (!fin.good()) break;
    fin.seekg(len, std::ios_base::cur);
    if (!fin.good()) break;
    last_tuple_offset = fin.tellg();
  }
  fin.close();
  return last_tuple_offset;
}
}  // namespace io
}  // namespace singa
