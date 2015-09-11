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
#include "utils/data_shard.h"

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

using namespace singa;

TEST(DataShardTest, CreateDataShard) {
  std::string path = "src/test/shard_test";
  mkdir(path.c_str(), 0755);
  DataShard shard(path, DataShard::kCreate, 50);
  shard.Insert(key[0], tuple[0]);
  shard.Insert(key[1], tuple[1]);
  shard.Insert(key[2], tuple[2]);
  shard.Flush();
}

TEST(DataShardTest, AppendDataShard) {
  std::string path = "src/test/shard_test";
  DataShard shard(path, DataShard::kAppend, 50);
  shard.Insert(key[3], tuple[3]);
  shard.Insert(key[4], tuple[4]);
  shard.Flush();
}

TEST(DataShardTest, CountDataShard) {
  std::string path = "src/test/shard_test";
  DataShard shard(path, DataShard::kRead, 50);
  int count = shard.Count();
  ASSERT_EQ(5, count);
}

TEST(DataShardTest, ReadDataShard) {
  std::string path = "src/test/shard_test";
  DataShard shard(path, DataShard::kRead, 50);
  std::string k, t;
  ASSERT_TRUE(shard.Next(&k, &t));
  ASSERT_STREQ(key[0].c_str(), k.c_str());
  ASSERT_STREQ(tuple[0].c_str(), t.c_str());
  ASSERT_TRUE(shard.Next(&k, &t));
  ASSERT_STREQ(key[1].c_str(), k.c_str());
  ASSERT_STREQ(tuple[1].c_str(), t.c_str());
  ASSERT_TRUE(shard.Next(&k, &t));
  ASSERT_TRUE(shard.Next(&k, &t));
  ASSERT_TRUE(shard.Next(&k, &t));
  ASSERT_STREQ(key[4].c_str(), k.c_str());
  ASSERT_STREQ(tuple[4].c_str(), t.c_str());
  ASSERT_FALSE(shard.Next(&k, &t));
  shard.SeekToFirst();
  ASSERT_TRUE(shard.Next(&k, &t));
  ASSERT_STREQ(key[0].c_str(), k.c_str());
  ASSERT_STREQ(tuple[0].c_str(), t.c_str());
}
