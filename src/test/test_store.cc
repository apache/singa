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
#include <string>
#include "gtest/gtest.h"
#include "singa/io/store.h"

TEST(TextFileStore, Open) {
  auto store = singa::io::CreateStore("textfile");
  EXPECT_EQ(store->Open("src/test/store.txt", singa::io::kCreate), true);
  store->Close();
  EXPECT_EQ(store->Open("src/test/store.txt", singa::io::kRead), true);
  store->Close();
}

TEST(TextFileStore, Write) {
  auto store = singa::io::CreateStore("textfile");
  store->Open("src/test/store.txt", singa::io::kCreate);
  store->Write("001", "first tuple");
  store->Write("002", "second tuple");
  store->Flush();
  store->Write("003", "third tuple");
  store->Close();
}

TEST(TextFileStore, Read) {
  auto store = singa::io::CreateStore("textfile");
  EXPECT_EQ(store->Open("src/test/store.txt", singa::io::kRead), true);
  std::string key, value;
  EXPECT_EQ(store->Read(&key, &value), true);
  EXPECT_EQ(key, "0");
  EXPECT_EQ(value, "first tuple");

  EXPECT_EQ(store->Read(&key, &value), true);
  EXPECT_EQ(store->Read(&key, &value), true);
  EXPECT_EQ(store->Read(&key, &value), false);
  store->SeekToFirst();

  EXPECT_EQ(store->Read(&key, &value), true);
  EXPECT_EQ(key, "0");
  EXPECT_EQ(value, "first tuple");
}
TEST(KVFileStore, Open) {
  auto store = singa::io::CreateStore("kvfile");
  EXPECT_EQ(store->Open("src/test/store.bin", singa::io::kCreate), true);
  store->Close();
  EXPECT_EQ(store->Open("src/test/store.bin", singa::io::kRead), true);
  store->Close();
}
TEST(KVFileStore, Write) {
  auto store = singa::io::CreateStore("kvfile");
  store->Open("src/test/store.bin", singa::io::kCreate);
  store->Write("001", "first tuple");
  store->Write("002", "second tuple");
  store->Flush();
  store->Write("003", "third tuple");
  store->Close();
}
TEST(KVFileStore, Read) {
  auto store = singa::io::CreateStore("kvfile");
  store->Open("src/test/store.bin", singa::io::kRead);
  std::string key, value;
  EXPECT_EQ(store->Read(&key, &value), true);
  EXPECT_EQ(key, "001");
  EXPECT_EQ(value, "first tuple");

  EXPECT_EQ(store->Read(&key, &value), true);
  EXPECT_EQ(store->Read(&key, &value), true);
  EXPECT_EQ(store->Read(&key, &value), false);
  store->SeekToFirst();

  EXPECT_EQ(store->Read(&key, &value), true);
  EXPECT_EQ(key, "001");
  EXPECT_EQ(value, "first tuple");
}
