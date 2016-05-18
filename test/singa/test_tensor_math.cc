#include "gtest/gtest.h"
#include "singa/core/tensor.h"
using singa::Tensor;
using singa::Shape;
using singa::Device;

class TestTensorMath : public ::testing::Test {
 protected:
  virtual void SetUp() {
    const float dat1[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float dat2[] = {1.1f, 2.1f, 3.1f, 4.1f, 5.1f, 6.1f};
    a.ReShape(singa::Shape{6});
    b.ReShape(singa::Shape{6});
    c.ReShape(singa::Shape{6, 1});
    d.ReShape(singa::Shape{3, 2});

    a.CopyDataFromHostPtr<float>(dat1, 6);
    b.CopyDataFromHostPtr<float>(dat2, 6);
  }
  Tensor a, b, c, d;
};

TEST_F(TestTensorMath, MemberAddTensor) {
  Tensor aa = a.Clone();
  aa += a;
  const float* dptr = aa.data<const float*>();
  EXPECT_FLOAT_EQ(2.0f, dptr[0]);
  EXPECT_FLOAT_EQ(4.0f, dptr[1]);
  EXPECT_FLOAT_EQ(6.0f, dptr[2]);

  // check p is initialized to 0
  Tensor p(Shape{6});
  p += aa;
  const float* dptr1 = p.data<const float*>();
  EXPECT_FLOAT_EQ(2.0f, dptr1[0]);
  EXPECT_FLOAT_EQ(4.0f, dptr1[1]);
  EXPECT_FLOAT_EQ(6.0f, dptr1[2]);

  a += b;
  const float* dptr2 = a.data<const float*>();
  EXPECT_FLOAT_EQ(2.1f, dptr2[0]);
  EXPECT_FLOAT_EQ(4.1f, dptr2[1]);
  EXPECT_FLOAT_EQ(6.1f, dptr2[2]);
  EXPECT_FLOAT_EQ(12.1f, dptr2[5]);
}


TEST_F(TestTensorMath, AddTensors) {
  Tensor ret(a.shape(), a.device(), a.data_type());
  Add(a, b, &ret);
  const float* dptr = ret.data<const float*>();
  EXPECT_FLOAT_EQ(2.1f, dptr[0]);
  EXPECT_FLOAT_EQ(4.1f, dptr[1]);
  EXPECT_FLOAT_EQ(6.1f, dptr[2]);
  EXPECT_FLOAT_EQ(12.1f, dptr[5]);

  const Tensor d = a + b;
  const float* dptr2 = d.data<const float*>();
  EXPECT_FLOAT_EQ(2.1f, dptr2[0]);
  EXPECT_FLOAT_EQ(4.1f, dptr2[1]);
  EXPECT_FLOAT_EQ(6.1f, dptr2[2]);
  EXPECT_FLOAT_EQ(12.1f, dptr2[5]);

  Add(a, b, &a);
  const float* dptr1 = a.data<const float*>();
  EXPECT_FLOAT_EQ(2.1f, dptr1[0]);
  EXPECT_FLOAT_EQ(4.1f, dptr1[1]);
  EXPECT_FLOAT_EQ(6.1f, dptr1[2]);
  EXPECT_FLOAT_EQ(12.1f, dptr1[5]);
}
