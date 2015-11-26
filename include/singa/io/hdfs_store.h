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

#ifndef SINGA_IO_HDFS_STORE_H_
#define SINGA_IO_HDFS_STORE_H_

#include <string>
#include "singa/io/store.h"
#include "singa/io/hdfsfile.h"

namespace singa {
namespace io {

/**
 * HDFS implementation of the Store interface. The store manages key-value 
 * records storing in HDFS files. 
 *
 * The store consists of records of the following format:
 *      [<length><content>] 
 */
class HDFSStore : public Store {
 public:
  ~HDFSStore() { Close();}
  bool Open(const std::string& source, Mode mode) override;
  void Close() override;
  bool Read(std::string* key, std::string* value) override;
  void SeekToFirst() override;
  void Seek(int offset) override;
  bool Write(const std::string& key, const std::string& value) override;
  void Flush() override;

 private:
  HDFSFile* file_ = nullptr;
  Mode mode_;
};

}  // namespace io
}  // namespace singa

#endif  // SINGA_IO_HDFS_STORE_H_
