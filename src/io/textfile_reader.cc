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
bool TextFileReader::Open(const std::string& path) {
  path_ = path;
  fdat_.open(path_, std::ios::in);
  if (!fdat_.is_open()) LOG(WARNING) << "Cannot open file " << path_;
  return fdat_.is_open();
}

void TextFileReader::Close() {
  if (fdat_.is_open()) fdat_.close();
}

bool TextFileReader::Read(std::string* key, std::string* value) {
  CHECK(fdat_.is_open()) << "File not open!";
  key->clear();
  value->clear();
  if (!std::getline(fdat_, *value)) {
    if (fdat_.eof())
      return false;
    else
      LOG(FATAL) << "Error in reading text file";
  }
  *key = std::to_string(lineNo_++);
  return true;
}

int TextFileReader::Count() {
  std::ifstream fin(path_, std::ios::in);
  CHECK(fin.is_open()) << "Cannot create file " << path_;
  int count = 0;
  string line;
  while (!fin.eof()) {
    std::getline(fin, line);
    if (line != "") count++;
  }
  fin.close();
  return count;
}

void TextFileReader::SeekToFirst() {
  CHECK(fdat_.is_open());
  lineNo_ = 0;
  fdat_.clear();
  fdat_.seekg(0);
}
}  // namespace io
}  // namespace singa
