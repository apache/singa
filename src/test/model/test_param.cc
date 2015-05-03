#include <gtest/gtest.h>
#include <glog/logging.h>
#include "proto/model.pb.h"

#include "utils/param.h"

using namespace singa;

class ParamTest : public ::testing::Test {
 public:
  ParamTest() {
    wp.set_name("weight");
    wp.add_shape(3);
    wp.add_shape(4);
    bp.set_name("bias");
    bp.add_shape(4);
  }
 protected:
  Param w, b;
  ParamProto wp, bp;
};

TEST_F(ParamTest, ConstantInit) {
  bp.set_init_method(ParamProto::kConstant);
  bp.set_value(0.5);
  b.Init(bp);
  const float *val = b.content().dptr;
  EXPECT_EQ(0.5f, val[0]);
  EXPECT_EQ(0.5f, val[1]);
  EXPECT_EQ(0.5f, val[2]);
  EXPECT_EQ(0.5f, val[3]);
  wp.set_init_method(ParamProto::kConstant);
  wp.set_value(1.5);
  w.Init(wp);
  val = w.content().dptr;
  EXPECT_EQ(1.5f, val[0]);
  EXPECT_EQ(1.5f, val[3]);
  EXPECT_EQ(1.5f, val[4]);
  EXPECT_EQ(1.5f, val[11]);
}

TEST_F(ParamTest, UniformInit) {
  bp.set_init_method(ParamProto::kUniform);
  bp.set_value(1.0f);
  b.Init(bp);
  const float *val = b.content().dptr;
  EXPECT_TRUE(val[0] >= -1 && val[0] <= 1);
  EXPECT_TRUE(val[1] >= -1 && val[2] <= 1);
  EXPECT_TRUE(val[2] >= -1 && val[2] <= 1);
  EXPECT_TRUE(val[3] >= -1 && val[3] <= 1);
  wp.set_init_method(ParamProto::kUniform);
  wp.set_value(1.0f);
  w.Init(wp);
  val = w.content().dptr;
  EXPECT_TRUE(val[0] >= -1 && val[0] <= 1);
  EXPECT_TRUE(val[3] >= -1 && val[3] <= 1);
  EXPECT_TRUE(val[4] >= -1 && val[4] <= 1);
  EXPECT_TRUE(val[11] >= -1 && val[11] <= 1);
}

TEST_F(ParamTest, UniformSqrtFanInInit) {
  wp.set_init_method(ParamProto::kUniformSqrtFanIn);
  wp.set_value(2.0f);
  w.Init(wp);
  const float *val = w.content().dptr;
  EXPECT_TRUE(val[0] >= -2 && val[0] <= 2);
  EXPECT_TRUE(val[3] >= -2 && val[3] <= 2);
  EXPECT_TRUE(val[4] >= -2 && val[4] <= 2);
  EXPECT_TRUE(val[11] >= -2 && val[11] <= 2);
}


TEST_F(ParamTest, UniformSqrtFanInOutInit) {
  wp.set_init_method(ParamProto::kUniformSqrtFanInOut);
  wp.set_value(1.0f);
  float low=1.0f, high=5.0f;
  wp.set_low(low);
  wp.set_high(high);
  w.Init(wp);
  const float *val = w.content().dptr;
  /*
  LOG(INFO) << val[0] << " " << val[1] << " " << val[2] << " " << val[3];
  LOG(INFO) << val[4] << " " << val[5] << " " << val[6] << " " << val[7];
  LOG(INFO) << val[8] << " " << val[9] << " " << val[10] << " " << val[11];
  */
  float factor = wp.value() / sqrt(wp.shape(0) + wp.shape(1));
  low=low*factor;
  high=high*factor;
  LOG(INFO)<<low<<" "<<high;
  EXPECT_TRUE(val[0] >= low && val[0] <= high);
  EXPECT_TRUE(val[3] >= low && val[3] <= high);
  EXPECT_TRUE(val[4] >= low && val[4] <= high);
  EXPECT_TRUE(val[11] >= low && val[11] <= high);
}

TEST_F(ParamTest, GaussianInit) {
  int len=5000, mean=0.0f, std=1.0f;
  ParamProto p;
  p.set_name("bias");
  p.add_shape(1);
  p.add_shape(len);
  p.set_init_method(ParamProto::kGaussain);
  p.set_value(1.0f);
  p.set_mean(mean);
  p.set_std(std);
  w.Init(p);

  const float *val = w.content().dptr;
  float dmean=0.0f;
  for(int i=0;i<len;i++)
    dmean+=val[i];
  dmean/=len;
  float dstd=0.0f;
  for(int i=0;i<len;i++)
    dstd+=(dmean-val[i])*(dmean-val[i]);
  dstd/=len;
  EXPECT_TRUE(std::abs(mean-dmean)<0.1);
  EXPECT_TRUE(std::abs(std-dstd)<0.1);
  /*
  LOG(INFO) << val[0] << " " << val[1] << " " << val[2] << " " << val[3];
  LOG(INFO) << val[4] << " " << val[5] << " " << val[6] << " " << val[7];
  LOG(INFO) << val[8] << " " << val[9] << " " << val[10] << " " << val[11];
  */
}

TEST_F(ParamTest, GaussianSqrtFanInInit) {
  wp.set_init_method(ParamProto::kGaussainSqrtFanIn);
  wp.set_value(1.0f);
  wp.set_mean(0);
  wp.set_std(1.0f);
  w.Init(wp);
  //const float *val = w.content().dptr;
  /*
  LOG(INFO) << val[0] << " " << val[1] << " " << val[2] << " " << val[3];
  LOG(INFO) << val[4] << " " << val[5] << " " << val[6] << " " << val[7];
  LOG(INFO) << val[8] << " " << val[9] << " " << val[10] << " " << val[11];
  */
}
