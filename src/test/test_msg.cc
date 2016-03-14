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

#include "gtest/gtest.h"
#include "singa/comm/msg.h"
using namespace singa;
TEST(MsgTest, AddrTest) {
  int src_grp = 1, src_worker = 2;
  int dst_grp = 0, dst_server = 1;
  int src_addr = Addr(src_grp, src_worker, 0);
  int dst_addr = Addr(dst_grp, dst_server, 1);
  Msg msg(src_addr, dst_addr);
  msg.set_trgt(123, -1);
  ASSERT_EQ(AddrGrp(msg.src()), src_grp);
  ASSERT_EQ(AddrID(msg.src()), src_worker);
  ASSERT_EQ(AddrType(msg.src()), 0);

  msg.SwapAddr();
  ASSERT_EQ(AddrGrp(msg.src()), dst_grp);
  ASSERT_EQ(AddrID(msg.src()), dst_server);
  ASSERT_EQ(AddrType(msg.src()), 1);
  ASSERT_EQ(msg.trgt_val(), 123);
  ASSERT_EQ(msg.trgt_version(), -1);
}

TEST(MsgTest, AddFrameTest) {
  int buf[5] = {1, 2, 3, 4, 5};
  Msg msg;
  msg.AddFrame("abcdefg", 7);
  msg.AddFrame(buf, sizeof(int) * 5);

  msg.FirstFrame();
  char* str = msg.FrameStr();
  ASSERT_STREQ(str, "abcdefg");
  delete str;
  ASSERT_EQ(msg.NextFrame(), true);
  int *val = static_cast<int*>(msg.FrameData());
  ASSERT_EQ(val[3], 4);
  ASSERT_EQ(msg.NextFrame(), false);

  msg.FirstFrame();
  str = msg.FrameStr();
  ASSERT_STREQ(str, "abcdefg");
  msg.LastFrame();
  val = static_cast<int*>(msg.FrameData());
  ASSERT_EQ(val[2], 3);
}

TEST(MsgTest, AddFormatFrame) {
  int x = 5;
  Msg msg;
  msg.AddFormatFrame("i", 12);
  msg.AddFormatFrame("f", 10.f);
  msg.AddFormatFrame("s", "abc");
  msg.AddFormatFrame("p", &x);
  msg.AddFormatFrame("isfp", 12, "abc", 10.f, &x);

  msg.FirstFrame();
  int y;
  msg.ParseFormatFrame("i", &y);
  ASSERT_EQ(y, 12);
  ASSERT_EQ(msg.NextFrame(), true);

  float z;
  msg.ParseFormatFrame("f", &z);
  ASSERT_EQ(z, 10.f);
  ASSERT_EQ(msg.NextFrame(), true);

  char buf[10];
  msg.ParseFormatFrame("s", buf);
  ASSERT_STREQ(buf, "abc");
  ASSERT_EQ(msg.NextFrame(), true);

  int *p;
  msg.ParseFormatFrame("p", &p);
  ASSERT_EQ(p, &x);
  ASSERT_EQ(msg.NextFrame(), true);

  msg.ParseFormatFrame("isfp", &y, buf, &z, &p);
  ASSERT_EQ(y, 12);
  ASSERT_STREQ(buf, "abc");
  ASSERT_EQ(z, 10.f);
  ASSERT_EQ(p, &x);
}
