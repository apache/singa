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
bool TextFileWriter::Open(const std::string& path, Mode mode) {
  CHECK(!fdat_.is_open());
  path_ = path;
  mode_ = mode;
  switch (mode) {
    case kCreate:
      fdat_.open(path_, std::ios::out | std::ios::trunc);
      CHECK(fdat_.is_open()) << "Cannot create file " << path_;
      break;
    case kAppend:
      fdat_.open(path_, std::ios::app);
      CHECK(fdat_.is_open()) << "Cannot open file " << path_;
      break;
    default:
      LOG(FATAL) << "unknown mode to open text file " << mode;
      break;
  }
  return fdat_.is_open();
}

void TextFileWriter::Close() {
  Flush();
  if (fdat_.is_open()) fdat_.close();
}

bool TextFileWriter::Write(const std::string& key, const std::string& value) {
  CHECK(fdat_.is_open()) << "File not open!";
  if (value.size() == 0) return false;
  fdat_ << value << std::endl;
  return true;
}

void TextFileWriter::Flush() {
  if (fdat_.is_open()) fdat_.flush();
}
}  // namespace io
}  // namespace singa
