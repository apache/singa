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

#ifndef SINGA_IO_WRITER_H_
#define SINGA_IO_WRITER_H_

#include <cstring>
#include <fstream>
#include <string>

#include "singa/singa_config.h"

#ifdef USE_LMDB
#include <lmdb.h>
#include <sys/stat.h>

#include <vector>
#endif  // USE_LMDB

namespace singa {
namespace io {

using std::string;
enum Mode { kCreate, kAppend };

/// General Writer that provides functions for writing tuples.
/// Subclasses implement the functions for a specific data storage, e.g., CSV
/// file, HDFS, image folder, leveldb, lmdb, etc.
class Writer {
 public:
  /// In case that users forget to call Close() to release resources, e.g.,
  /// memory, you can release them here.
  virtual ~Writer() {}

  /// Open a file.
  /// path is the path to the disk BinFile, it can be
  ///  - a path to local disk file.
  ///  - a path to local directory. This is to be compatible with the older
  ///    version (DataShard). The KVFile is shard.dat under that directory
  ///  - a hdfs file starting with "hdfs://"
  /// mode is open mode(kCreate, kAppend).
  virtual bool Open(const std::string &path, Mode mode) = 0;

  /// Release resources.
  virtual void Close() = 0;

  /// Write a key-value tuple.
  /// return true if success, otherwise false.
  virtual bool Write(const std::string &key, const std::string &value) = 0;

  /// Flush writing buffer if it has.
  virtual void Flush() = 0;
};

/// BinFile stores training/validation/test tuples.
/// Each tuple is encoded as [magic_word, key_len, key, val_len, val]:
///  - magic_word has 4 bytes; the first two are "s" and "g", the third one
/// indicates whether key is null, the last one is reserved for future use.
///  - key_len and val_len are of type uint32, which indicate the bytes of key
/// and value respectively;
///  - key_len and key are optional.)
/// When BinFile is created, it will remove the last tuple if the value size
/// and key size do not match because the last write crashed.
class BinFileWriter : public Writer {
 public:
  ~BinFileWriter() { Close(); }
  /// \copydoc Open(const std::string &path, Mode mode)
  bool Open(const std::string &path, Mode mode) override;
  /// \copydoc Open(const std::string& path), user defines capacity
  bool Open(const std::string &path, Mode mode, int capacity);
  /// \copydoc Close()
  void Close() override;
  /// \copydoc Write(const std::string& key, const std::string& value) override;
  bool Write(const std::string &key, const std::string &value) override;
  /// \copydoc Flush()
  void Flush() override;
  /// return path to binary file
  inline std::string path() { return path_; }

 protected:
  /// Open a file with path_ and initialize buf_
  bool OpenFile();

 private:
  /// file to be written
  std::string path_ = "";
  Mode mode_;
  /// ofstream
  std::ofstream fdat_;
  /// internal buffer
  char *buf_ = nullptr;
  /// allocated bytes for the buf_
  int capacity_ = 10485760;
  /// bytes in buf_
  int bufsize_ = 0;
  /// magic word
  const char kMagicWord[2] = {'s', 'g'};
};

/// TextFileWriter write training/validation/test tuples in CSV file.
class TextFileWriter : public Writer {
 public:
  ~TextFileWriter() { Close(); }
  /// \copydoc Open(const std::string &path, Mode mode)
  bool Open(const std::string &path, Mode mode) override;
  /// \copydoc Close()
  void Close() override;
  /// \copydoc Write(const std::string& key, const std::string& value) override;
  bool Write(const std::string &key, const std::string &value) override;
  /// \copydoc Flush()
  void Flush() override;
  /// return path to text file
  inline std::string path() { return path_; }

 private:
  /// file to be written
  std::string path_ = "";
  Mode mode_;
  /// ofstream
  std::ofstream fdat_;
};

#ifdef USE_LMDB
/// LMDBWriter write training/validation/test tuples into LMDB.
class LMDBWriter : public Writer {
 public:
  ~LMDBWriter() { Close(); }
  /// \copydoc Open(const std::string &path, Mode mode)
  bool Open(const std::string &path, Mode mode) override;
  /// \copydoc Close()
  void Close() override;
  /// \copydoc Write(const std::string& key, const std::string& value) override;
  bool Write(const std::string &key, const std::string &value) override;
  /// \copydoc Flush()
  void Flush() override;
  /// return path to text file
  inline std::string path() { return path_; }

 protected:
  void DoubleMapSize();
  inline void MDB_CHECK(int mdb_status);

 private:
  /// file to be written
  std::string path_ = "";
  /// kCreate or kAppend
  Mode mode_;
  /// lmdb env variable
  MDB_env *mdb_env_ = nullptr;
  /// buffer for key-value pairs
  std::vector<string> keys, values;
};
#endif  // USE_LMDB

}  // namespace io
}  // namespace singa

#endif  // SINGA_IO_WRITER_H_
