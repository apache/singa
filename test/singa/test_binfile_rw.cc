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

#include "../include/singa/io/reader.h"
#include "../include/singa/io/writer.h"
#include "gtest/gtest.h"

const char* path = "./binfile_test";
using singa::io::BinFileReader;
using singa::io::BinFileWriter;
TEST(BinFileWriter, Create) {
  BinFileWriter writer;
  bool ret;
  ret = writer.Open(path, singa::io::kCreate);
  EXPECT_EQ(true, ret);

  std::string key = "";
  std::string value = "\nThis is a test for binfile io.";
  ret = writer.Write(key, value);
  EXPECT_EQ(true, ret);

  ret = writer.Write(key, value);
  EXPECT_EQ(true, ret);

  writer.Flush();
  writer.Close();
}

TEST(BinFileWriter, Append) {
  BinFileWriter writer;
  bool ret;
  ret = writer.Open(path, singa::io::kAppend);
  EXPECT_EQ(true, ret);

  std::string key = "1";
  std::string value = "\nThis is another test for binfile io.";
  ret = writer.Write(key, value);
  EXPECT_EQ(true, ret);

  key = "2";
  value = "\nThis is another test for binfile io.";
  ret = writer.Write(key, value);
  EXPECT_EQ(true, ret);

  writer.Flush();
  writer.Close();
}

TEST(BinFileReader, Read) {
  BinFileReader reader;
  bool ret;
  ret = reader.Open(path);
  EXPECT_EQ(true, ret);

  int cnt = reader.Count();
  EXPECT_EQ(4, cnt);

  std::string key, value;
  reader.Read(&key, &value);
  EXPECT_STREQ("", key.c_str());
  EXPECT_STREQ("\nThis is a test for binfile io.", value.c_str());

  reader.Read(&key, &value);
  EXPECT_STREQ("", key.c_str());
  EXPECT_STREQ("\nThis is a test for binfile io.", value.c_str());

  reader.Read(&key, &value);
  EXPECT_STREQ("1", key.c_str());
  EXPECT_STREQ("\nThis is another test for binfile io.", value.c_str());

  reader.Read(&key, &value);
  EXPECT_STREQ("2", key.c_str());
  EXPECT_STREQ("\nThis is another test for binfile io.", value.c_str());

  reader.Close();
  remove(path);
}
