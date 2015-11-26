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

* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/

#include "singa/io/hdfsfile.h"

#include <glog/logging.h>
#include <iostream>
namespace singa {
namespace io {

HDFSFile::HDFSFile(const std::string& path, Mode mode): path_(path),
  mode_(mode) {
  // check that path starts with hdfs://
  CHECK_EQ(path.find("hdfs://"), 0);

  // extract namenode from path
  int path_idx = path.find_first_of("/", 7);
  int colon_idx = path.find_first_of(":", 7);
  std::string namenode = path.substr(7, colon_idx-7);
  int port = atoi(path.substr(colon_idx+1, path_idx-colon_idx-1).c_str());
  std::string filepath = path.substr(path_idx);

  // connect to HDFS
  fs_ = hdfsConnect(namenode.c_str(), port);
  CHECK_NOTNULL(fs_);

  if (mode == HDFSFile::kRead) {
    file_ = hdfsOpenFile(fs_, filepath.c_str(), O_RDONLY, 0, 0, 0);
  } else {
    // check if the directory exists, create it if not.
    int file_idx = path.find_last_of("/");
    std::string hdfs_directory_path = path.substr(path_idx, file_idx-path_idx);
    if (hdfsExists(fs_, hdfs_directory_path.c_str()) == -1)
      CHECK_EQ(hdfsCreateDirectory(fs_, hdfs_directory_path.c_str()), 0);
    file_ = hdfsOpenFile(fs_, filepath.c_str(), O_WRONLY, 0, 0, 0);
  }

  CHECK_NOTNULL(file_);
}

HDFSFile::~HDFSFile() {
  if (mode_ != HDFSFile::kRead)
    Flush();
  hdfsCloseFile(fs_, file_);
}

#ifdef USE_PROTOBUF
bool HDFSFile::Next(google::protobuf::Message* val) {
  // read from file_, then turns it to a message
  // red size, then content
  int size;
  if (hdfsRead(fs_, file_, &size, sizeof(int)) <= 0)
    return false;
  char *temp_buf = reinterpret_cast<char*>(malloc(size*sizeof(char)));
  CHECK(hdfsRead(fs_, file_, temp_buf, size));
  val->ParseFromArray(temp_buf, size);
  free(temp_buf);
  return true;
}

bool HDFSFile::Insert(const google::protobuf::Message& val) {
  std::string str;
  val.SerializeToString(&str);
  return Insert(str);
}
#endif

bool HDFSFile::Next(std::string* val) {
  char size_buf[sizeof(int)];
  // a hack to read across blocks. The first read my return in complete data,
  // so try the second read.
  int read_size_size = hdfsRead(fs_, file_, size_buf, sizeof(int));

  if (read_size_size == 0)
    return false;

  if (read_size_size < (static_cast<int>(sizeof(int))))
    CHECK_EQ(hdfsRead(fs_, file_, size_buf+read_size_size,
      sizeof(int)-read_size_size),
      sizeof(int)-read_size_size);
  int size;
  memcpy(&size, size_buf, sizeof(int));

  char *temp_buf = reinterpret_cast<char*>(malloc(size*sizeof(char)));

  int read_size = hdfsRead(fs_, file_, temp_buf, size);
  if (read_size < size)
    CHECK_EQ(hdfsRead(fs_, file_, temp_buf+read_size, size-read_size),
      size-read_size);
  val->clear();
  val->append(temp_buf, size);
  free(temp_buf);
  return true;
}

// append one record to the end of the file
bool HDFSFile::Insert(const std::string& val) {
  CHECK(mode_ != HDFSFile::kRead);
  // write length, then content
  int size = val.length();
  CHECK_EQ(hdfsWrite(fs_, file_, &size, sizeof(int)), sizeof(int));
  CHECK_EQ(hdfsWrite(fs_, file_, val.c_str(), val.length()), val.length());
  return true;
}

void HDFSFile::Seek(int offset) {
  CHECK_EQ(mode_, kRead);
  // seek back to the parition offset
  CHECK_EQ(hdfsSeek(fs_, file_, offset), 0);
}

void HDFSFile::Flush() {
  CHECK_EQ(hdfsFlush(fs_, file_), 0);
}

}  // namespace io
}  // namespace singa
