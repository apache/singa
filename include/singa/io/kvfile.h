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

#ifndef SINGA_IO_KVFILE_H_
#define SINGA_IO_KVFILE_H_

#include <fstream>
#include <string>
#include <unordered_set>

#define USE_PROTOBUF 1

#ifdef USE_PROTOBUF
#include <google/protobuf/message.h>
#endif

namespace singa {
namespace io {

/**
 * KVFile stores training/validation/test tuples.
 * Every worker node should have a KVFile for training data (validation/test
 * KVFile is optional).
 * KVFile consists of a set of unordered tuples. Each tuple is
 * encoded as [key_len key val_len val] (key_len and val_len are of type
 * uint32, which indicate the bytes of key and value respectively.
 *
 * When KVFile is created, it will remove the last tuple if the value size
 * and key size do not match because the last write crashed.
 *
 * TODO(wangwei) split one KVFile into multiple KVFile s.
 *
 */
class KVFile {
 public:
  enum Mode {
    // read only mode used in training
    kRead = 0,
    // write mode used in creating KVFile (will overwrite previous one)
    kCreate = 1,
    // append mode, e.g. used when previous creating crashes
    kAppend = 2
  };

  /**
   * KVFile constructor.
   *
   * @param path path to the disk KVFile, it can be
   *  - a path to local disk file.
   *  - a path to local directory. This is to be compatible with the older
   *    version (DataShard). The KVFile is shard.dat under that directory
   *  - a hdfs file starting with "hdfs://"
   * @param mode KVFile open mode, KVFile::kRead, KVFile::kWrite or
   * KVFile::kAppend
   * @param bufsize Cache bufsize bytes data for every disk op (read or write),
   * default is 10MB.
   */
  KVFile(const std::string& path, Mode mode, int bufsize = 10485760);
  ~KVFile();

#ifdef USE_PROTOBUF
  /**
   * read next tuple from the KVFile.
   *
   * @param key Tuple key
   * @param val Record of type Message
   * @return false if read unsuccess, e.g., the tuple was not inserted
   *         completely.
   */
  bool Next(std::string* key, google::protobuf::Message* val);
  /**
   * Append one tuple to the KVFile.
   *
   * @param key e.g., image path
   * @param val
   * @return false if unsucess, e.g., inserted before
   */
  bool Insert(const std::string& key, const google::protobuf::Message& tuple);
#endif
  /**
   * read next tuple from the KVFile.
   *
   * @param key Tuple key
   * @param val Record of type string
   * @return false if unsuccess, e.g. the tuple was not inserted completely.
   */
  bool Next(std::string* key, std::string* val);
  /**
   * Append one tuple to the KVFile.
   *
   * @param key e.g., image path
   * @param val
   * @return false if unsucess, e.g., inserted before
   */
  bool Insert(const std::string& key, const std::string& tuple);
  /**
   * Move the read pointer to the head of the KVFile file.
   * Used for repeated reading.
   */
  void SeekToFirst();
  /**
   * Flush buffered data to disk.
   * Used only for kCreate or kAppend.
   */
  void Flush();
  /**
   * Iterate through all tuples to get the num of all tuples.
   *
   * @return num of tuples
   */
  int Count();
  /**
   * @return path to KVFile file
   */
  inline std::string path() { return path_; }

 protected:
  /**
   * Read the next key and prepare buffer for reading value.
   *
   * @param key
   * @return length (i.e., bytes) of value field.
   */
  int Next(std::string* key);
  /**
   * Setup the disk pointer to the right position for append in case that
   * the pervious write crashes.
   *
   * @param path KVFile path.
   * @return offset (end pos) of the last success written record.
   */
  int PrepareForAppend(const std::string& path);
  /**
   * Read data from disk if the current data in the buffer is not a full field.
   *
   * @param size size of the next field.
   */
  bool PrepareNextField(int size);

 private:
  std::string path_ = "";
  Mode mode_;
  //!< either ifstream or ofstream
  std::fstream fdat_;
  //!< to avoid replicated record
  std::unordered_set<std::string> keys_;
  //!< internal buffer
  char* buf_ = nullptr;
  //!< offset inside the buf_
  int offset_ = 0;
  //!< allocated bytes for the buf_
  int capacity_ = 0;
  //!< bytes in buf_, used in reading
  int bufsize_ = 0;
};
}  // namespace io

/**
 * @deprecated {ShardData is deprecated! Use KVFile}.
 */
using DataShard = io::KVFile;
}  // namespace singa

#endif  // SINGA_IO_KVFILE_H_
