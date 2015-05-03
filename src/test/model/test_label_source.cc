// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-21 19:40

#include <gtest/gtest.h>
#include <glog/logging.h>
#include "proto/model.pb.h"
#include "disk/label_source.h"

namespace lapis {
class LabelSourceTest : public ::testing::Test {
 public:
  LabelSourceTest() {
    DataSourceProto ds;
    ds.set_path("src/test/data/label_source.dat");
    ds.set_size(12);
    ds.set_name("label source");
    ls.Init(ds);
  }

 protected:
  LabelSource ls;
};

TEST_F(LabelSourceTest, LoadData) {
  auto ptr2names = ls.LoadData(nullptr);
  EXPECT_EQ(12, ptr2names->size());
  EXPECT_STREQ("img0.JPEG", ptr2names->at(0).c_str());
  EXPECT_STREQ("img1.JPEG", ptr2names->at(1).c_str());
  EXPECT_STREQ("img5.JPEG", ptr2names->at(5).c_str());
  EXPECT_STREQ("img10.JPEG", ptr2names->at(10).c_str());
  EXPECT_STREQ("img11.JPEG", ptr2names->at(11).c_str());
}

TEST_F(LabelSourceTest, GetData) {
  ls.LoadData(nullptr);
  Blob b;
  b.Resize(1, 1, 1, 5);
  ls.GetData(&b);
  const float *val = b.dptr;
  EXPECT_EQ(0.0f, val[0]);
  EXPECT_EQ(1.0f, val[1]);
  EXPECT_EQ(4.0f, val[2]);
  EXPECT_EQ(9.0f, val[3]);
  EXPECT_EQ(16.0f, val[4]);
  ls.GetData(&b);
  EXPECT_EQ(4.0f, val[0]);
  EXPECT_EQ(5.0f, val[1]);
  EXPECT_EQ(6.0f, val[2]);
  EXPECT_EQ(7.0f, val[3]);
  EXPECT_EQ(8.0f, val[4]);
  ls.GetData(&b);
  EXPECT_EQ(1.0f, val[0]);
  EXPECT_EQ(2.0f, val[1]);
  EXPECT_EQ(0.0f, val[2]);
  EXPECT_EQ(1.0f, val[3]);
  EXPECT_EQ(4.0f, val[4]);
}

}  // namespace lapis
