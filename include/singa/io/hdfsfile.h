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

#ifndef SINGA_IO_HDFSFILE_H_
#define SINGA_IO_HDFSFILE_H_

#include <fstream>
#include <string>
#include <unordered_set>


#define USE_PROTOBUF 1

#ifdef USE_PROTOBUF
#include <google/protobuf/message.h>
#endif

#include <hdfs/hdfs.h>

namespace singa {
namespace io {

/**
 * HDFSFile represents a specific partition of the HDFS file storing training/validation
 * or test data. HDFS library maintains its own buffer, so we don't need one. 
 * 
 * Each record is of the form: <length><content>
 */
class HDFSFile {
 public:
  enum Mode {
    // read only mode used in training
    kRead = 0,
    // write mode used in creating HDFSFile (will overwrite previous one)
    kCreate = 1,
    // append mode, e.g. used when previous creating crashes
    kAppend = 2
  };

  /**
   * HDFSFile constructor.
   *
   * @param path path to file, of the form "hdfs://namenode/file_path"
   * @param mode HDFSFile::kRead, HDFSFile::kCreate or HDFSFile::kAppend
   */
  HDFSFile(const std::string& path, Mode mode);
  ~HDFSFile();

#ifdef USE_PROTOBUF
  /**
   * read next tuple from the HDFSFile.
   *
   * @param val Record of type Message
   * @return false if read unsuccess, e.g., the tuple was not inserted
   *         completely.
   */
  bool Next(google::protobuf::Message* val);
  /**
   * Append one record to the HDFSFile.
   *
   * @param val
   * @return false if unsucess, e.g., inserted before
   */
  bool Insert(const google::protobuf::Message& tuple);
#endif

  /**
   * Read next record from the HDFSFile.
   *
   * @param val Record of type string
   * @return false if unsuccess, e.g. the tuple was not inserted completely.
   */
  bool Next(std::string* val);
  /**
   * Append record to the KVFile.
   *
   * @param key e.g., image path
   * @param val
   * @return false if unsucess, e.g., inserted before
   */
  bool Insert(const std::string& tuple);
  /**
   * Move the read pointer to the head of the KVFile file.
   * Used for repeated reading.
   */
  void Seek(int offset);

  /**
   * Flush buffered data to disk.
   * Used only for kCreate or kAppend.
   */
  void Flush();
    /**
   * @return path to HDFSFile file
   */
  inline std::string path() { return path_; }

 private:
  std::string path_ = "";
  Mode mode_;
  // handle to HDFS
  hdfsFS fs_;
  // handle to the HDFS open file
  hdfsFile file_;

  //!< to avoid replicated record
  std::unordered_set<std::string> keys_;
};
}  // namespace io

}  // namespace singa

#endif  // SINGA_IO_HDFSFILE_H_
