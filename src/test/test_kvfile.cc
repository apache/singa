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

#include <sys/stat.h>

#include "gtest/gtest.h"
#include "singa/io/kvfile.h"

std::string key[] = {"firstkey",
                     "secondkey",
                     "3key",
                     "key4",
                     "key5"};
std::string tuple[] = {"firsttuple",
                       "2th-tuple",
                       "thridtuple",
                       "tuple4",
                       "tuple5"};
namespace singa {
namespace io {
TEST(KVFileTest, CreateKVFile) {
  std::string path = "src/test/kvfile.bin";
  KVFile kvfile(path, KVFile::kCreate, 50);
  kvfile.Insert(key[0], tuple[0]);
  kvfile.Insert(key[1], tuple[1]);
  kvfile.Insert(key[2], tuple[2]);
  kvfile.Flush();
}

TEST(KVFileTest, AppendKVFile) {
  std::string path = "src/test/kvfile.bin";
  KVFile kvfile(path, KVFile::kAppend, 50);
  kvfile.Insert(key[3], tuple[3]);
  kvfile.Insert(key[4], tuple[4]);
  kvfile.Flush();
}

TEST(KVFileTest, CountKVFile) {
  std::string path = "src/test/kvfile.bin";
  KVFile kvfile(path, KVFile::kRead, 50);
  int count = kvfile.Count();
  ASSERT_EQ(5, count);
}

TEST(KVFileTest, ReadKVFile) {
  std::string path = "src/test/kvfile.bin";
  KVFile kvfile(path, KVFile::kRead, 50);
  std::string k, t;
  ASSERT_TRUE(kvfile.Next(&k, &t));
  ASSERT_STREQ(key[0].c_str(), k.c_str());
  ASSERT_STREQ(tuple[0].c_str(), t.c_str());
  ASSERT_TRUE(kvfile.Next(&k, &t));
  ASSERT_STREQ(key[1].c_str(), k.c_str());
  ASSERT_STREQ(tuple[1].c_str(), t.c_str());
  ASSERT_TRUE(kvfile.Next(&k, &t));
  ASSERT_TRUE(kvfile.Next(&k, &t));
  ASSERT_TRUE(kvfile.Next(&k, &t));
  ASSERT_STREQ(key[4].c_str(), k.c_str());
  ASSERT_STREQ(tuple[4].c_str(), t.c_str());
  ASSERT_FALSE(kvfile.Next(&k, &t));
  kvfile.SeekToFirst();
  ASSERT_TRUE(kvfile.Next(&k, &t));
  ASSERT_STREQ(key[0].c_str(), k.c_str());
  ASSERT_STREQ(tuple[0].c_str(), t.c_str());
}
}  // namespace io
}  // namespace singa
