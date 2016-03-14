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

#ifndef SINGA_IO_STORE_H_
#define SINGA_IO_STORE_H_

#include <string>

namespace singa {
namespace io {

using std::string;
enum Mode { kCreate, kRead, kAppend };

/**
 * General key-value store that provides functions for reading and writing
 * tuples.
 *
 * Subclasses implement the functions for a specific data storage, e.g., CSV
 * file, HDFS, image folder, singa::io::SFile, leveldb, lmdb, etc.
 */
class Store {
 public:
  Store() { }
  /**
   * In case that users forget to call Close() to release resources, e.g.,
   * memory, you can release them here.
   */
  virtual ~Store() { }
  /**
   * @param[in] source path to the storage, could be a file path, folder path
   * or hdfs path, or even a http url.
   * @param[in] mode
   * @return true if open successfully, otherwise false.
   */
  virtual bool Open(const std::string& source, Mode mode) = 0;
  /**
   * Release resources.
   */
  virtual void Close() = 0;
  /**
   * Read a tuple.
   *
   * @param[out] key
   * @param[out] value
   * @return true if read successfully, otherwise false.
   */
  virtual bool Read(std::string* key, std::string* value) = 0;
  /**
   * Seek the read header to the first tuple.
   */
  virtual void SeekToFirst() = 0;

  /**
   * Seek to an offset. This allows concurrent workers to start reading from
   * different positions (HDFS). 
   */
  virtual void Seek(int offset) = 0; 
  /**
   * Write a tuple.
   *
   * @param[in] key
   * @param[in] value
   * @return true if success, otherwise false.
   */
  virtual bool Write(const std::string& key, const std::string& value) = 0;
  /**
   * Flush writing buffer if it has.
   */
  virtual void Flush() {}
};

/**
 * Create a Store object.
 *
 * @param[in] backend identifier for a specific backend. Two backends are
 * inluced currently, i.e., "kvfile", "textfile"
 * @return a pointer to the newly created Store.
 */
Store* CreateStore(const string& backend);
/**
 * Create and open a Store object.
 *
 * @param[in] backend, @see CreateStore().
 * @param[in] path
 * @param[in] mode kRead or kCreate or kAppend
 */
Store* OpenStore(const string& backend, const string& path, Mode mode);

}  // namespace io
}  // namespace singa

#endif  // SINGA_IO_STORE_H_
