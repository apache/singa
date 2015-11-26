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

#include <glog/logging.h>
#include "singa/io/hdfs_store.h"

namespace singa {
namespace io {

bool HDFSStore::Open(const std::string& source, Mode mode) {
  CHECK(file_ == nullptr);
  if (mode == kRead)
    file_ = new HDFSFile(source, HDFSFile::kRead);
  else if (mode == kCreate)
    file_ = new HDFSFile(source, HDFSFile::kCreate);
  else if (mode == kAppend)
    file_ = new HDFSFile(source, HDFSFile::kAppend);
  mode_ = mode;
  return file_ != nullptr;
}

void HDFSStore::Close() {
  if (file_ != nullptr)
    delete file_;
  file_ = nullptr;
}

bool HDFSStore::Read(std::string* key, std::string* value) {
  CHECK_EQ(mode_, kRead);
  CHECK(file_ != nullptr);
  return file_->Next(value);
}

void HDFSStore::SeekToFirst() {
  CHECK_EQ(mode_, kRead);
  CHECK(file_ != nullptr);
  file_->Seek(0);
}

void HDFSStore::Seek(int offset) {
  file_->Seek(offset);
}

bool HDFSStore::Write(const std::string& key, const std::string& value) {
  CHECK_NE(mode_, kRead);
  CHECK(file_ != nullptr);
  return file_->Insert(value);
}

void HDFSStore::Flush() {
  CHECK_NE(mode_, kRead);
  CHECK(file_!= nullptr);
  file_->Flush();
}

}  // namespace io
}  // namespace singa
