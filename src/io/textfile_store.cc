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


#include "singa/io/textfile_store.h"
#include <glog/logging.h>

namespace singa {
namespace io {

bool TextFileStore::Open(const std::string& source, Mode mode) {
  if (mode == kRead)
    fs_ = new std::fstream(source, std::fstream::in);
  else if (mode == kCreate)
    fs_ = new std::fstream(source, std::fstream::out);
  mode_ = mode;
  return fs_->is_open();
}

void TextFileStore::Close() {
  if (fs_ != nullptr) {
    if (fs_->is_open()) {
      if (mode_ != kRead)
        fs_->flush();
      fs_->close();
    }
    delete fs_;
    fs_ = nullptr;
  }
}

bool TextFileStore::Read(std::string* key, std::string* value) {
  CHECK_EQ(mode_, kRead);
  CHECK(fs_ != nullptr);
  CHECK(value != nullptr);
  CHECK(key != nullptr);
  if (!std::getline(*fs_, *value)) {
    if (fs_->eof())
      return false;
    else
      LOG(FATAL) << "error in reading csv file";
  }
  *key = std::to_string(lineNo_++);
  return true;
}

void TextFileStore::SeekToFirst() {
  CHECK_EQ(mode_, kRead);
  CHECK(fs_ != nullptr);
  lineNo_ = 0;
  fs_->clear();
  fs_->seekg(0);
}

void TextFileStore::Seek(int offset) {
}

bool TextFileStore::Write(const std::string& key, const std::string& value) {
  CHECK_NE(mode_, kRead);
  CHECK(fs_ != nullptr);
  // csv store does not write key
  *fs_ << value << '\n';
  return true;
}

void TextFileStore::Flush() {
  fs_->flush();
}

}  // namespace io
}  // namespace singa
