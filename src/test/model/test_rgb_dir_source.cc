// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-21 21:52

#include <gtest/gtest.h>
#include <glog/logging.h>
#include <algorithm>

#include "proto/model.pb.h"
#include "disk/rgb_dir_source.h"
#include "disk/label_source.h"

namespace lapis {
class RGBDirSourceTest : public ::testing::Test {
 public:
  RGBDirSourceTest() {
    DataSourceProto ds;
    ds.set_path("src/test/data/rgb_dir");
    ds.set_mean_file("src/test/data/imagenet_mean.binaryproto");
    ds.set_size(3);
    ds.set_height(256);
    ds.set_width(256);
    ds.set_offset(2);
    ds.set_name("rgb dir source");
    rgbs.Init(ds);
  }

 protected:
  RGBDirSource rgbs;
};

TEST_F(RGBDirSourceTest, LoadDataNoInputKeys) {
  auto &ptr2names = rgbs.LoadData(nullptr);
  EXPECT_EQ(3, ptr2names->size());
  sort(ptr2names->begin(), ptr2names->end());
  EXPECT_STREQ("img0.JPEG", ptr2names->at(0).c_str());
  EXPECT_STREQ("img1.JPEG", ptr2names->at(1).c_str());
  EXPECT_STREQ("img2.JPEG", ptr2names->at(2).c_str());
}

TEST_F(RGBDirSourceTest, LoadDataWithInputKeys) {
  LabelSource ls;
  DataSourceProto ds;
  ds.set_path("src/test/data/label_source.dat");
  ds.set_name("label source");
  ds.set_size(3);
  ls.Init(ds);
  auto ptr2names1 = ls.LoadData(nullptr);
  auto ptr2names2 = rgbs.LoadData(ptr2names1);
  EXPECT_EQ(3, ptr2names2->size());
  for (int i = 0; i < 3; i++)
    EXPECT_STREQ(ptr2names1->at(i).c_str(), ptr2names2->at(i).c_str());
}

TEST_F(RGBDirSourceTest, GetData) {
  Blob b;
  b.Resize(256,256,3,2);
  rgbs.LoadData(nullptr);
  rgbs.GetData(&b);
  rgbs.GetData(&b);
  rgbs.GetData(&b);
}
}  // namespace lapis

