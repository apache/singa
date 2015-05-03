// Copyright © 2014 Wei Wang. All Rights Reserved.
// 2014-07-18 19:44
#include <gtest/gtest.h>
#include "proto/model.pb.h"
#include "model/lapis.h"

namespace lapis {
class BlobTest : public ::testing::Test {
 public:
  BlobTest() : blob1(new Blob()), blob2(new Blob()) {}
  ~BlobTest() {
    delete blob1;
    delete blob2;
  }
 protected:
  Blob *blob1, *blob2;
  Blob blob3, blob4;
};

TEST_F(BlobTest, Constructor) {
  EXPECT_EQ(blob1->length(), 0);
  EXPECT_EQ(blob1->width(), 0);
  EXPECT_EQ(blob1->height(), 0);
  EXPECT_EQ(blob3.length(), 0);
  EXPECT_EQ(blob3.width(), 0);
  EXPECT_EQ(blob3.height(), 0);
  EXPECT_TRUE(blob2->dptr == nullptr);
  EXPECT_TRUE(blob4.dptr == nullptr);
}

TEST_F(BlobTest, TestResize) {
  blob1->Resize(10,1,1,1);
  EXPECT_EQ(blob1->length(), 10);
  EXPECT_EQ(blob1->num(), 10);
  EXPECT_EQ(blob1->height(), 1);
  EXPECT_EQ(blob1->width(), 1);
  EXPECT_TRUE(blob1->dptr != nullptr);
  blob2->Resize(4,1,1,3);
  EXPECT_EQ(blob2->length(), 12);
  EXPECT_EQ(blob2->num(), 4);
  EXPECT_EQ(blob2->height(), 1);
  EXPECT_EQ(blob2->width(), 3);
  EXPECT_TRUE(blob2->dptr != nullptr);
  blob3.Resize(5,1,4,3);
  EXPECT_EQ(blob3.length(), 60);
  EXPECT_EQ(blob3.num(), 5);
  EXPECT_EQ(blob3.height(), 4);
  EXPECT_EQ(blob3.width(), 3);
  EXPECT_TRUE(blob3.dptr != nullptr);
  blob4.Resize(6,5,4,3);
  EXPECT_EQ(blob4.length(), 360);
  EXPECT_EQ(blob4.num(), 6);
  EXPECT_EQ(blob4.height(), 4);
  EXPECT_EQ(blob4.width(), 3);
  EXPECT_TRUE(blob4.dptr != nullptr);
}

}  // namespace lapis
