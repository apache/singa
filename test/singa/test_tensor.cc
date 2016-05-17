#include "gtest/gtest.h"
#include "singa/core/tensor.h"
using singa::Tensor;
using singa::Shape;
using singa::Device;

TEST(TensorTest, TestConstructor) {
  singa::Tensor float_t(singa::Shape{2,3});
  EXPECT_EQ(6u, float_t.Size());
  EXPECT_EQ(sizeof(float) * 6, float_t.MemSize());
  EXPECT_EQ(singa::kFloat32, float_t.data_type());
  auto s = float_t.shape();
  EXPECT_EQ(s[0], 2u);
  EXPECT_EQ(s[1], 3u);

  EXPECT_NE(float_t.device(), nullptr);

  singa::Tensor float16_t(Shape{2,3}, singa::kFloat16);
  EXPECT_EQ(singa::kFloat16, float16_t.data_type());
  EXPECT_EQ(6u, float16_t.Size());
  EXPECT_EQ(12u, float16_t.blob()->size());

  singa::Tensor x(float16_t);
  EXPECT_EQ(float16_t.Size(), x.Size());
  EXPECT_EQ(float16_t.blob(), x.blob());
  EXPECT_EQ(float16_t.data_type(), x.data_type());
  EXPECT_EQ(float16_t.device(), x.device());

  singa::Tensor y = float16_t;
  EXPECT_EQ(float16_t.Size(), x.Size());
  EXPECT_EQ(float16_t.blob(), x.blob());
  EXPECT_EQ(float16_t.data_type(), x.data_type());
  EXPECT_EQ(float16_t.device(), x.device());
}

TEST(TensorClass, Reshape) {
  Tensor t;
  t.ReShape(Shape{2,3});
  EXPECT_TRUE((Shape{2,3} == t.shape()));

  t.ReShape(Shape{3,3, 4});
  EXPECT_TRUE((Shape{3,3, 4} == t.shape()));

  t.ReShape(Shape{12});
  EXPECT_TRUE((Shape{12} == t.shape()));

  Tensor o;
  EXPECT_TRUE(o.shape() != t.shape());
  o.ReShape(Shape{3, 3});
  EXPECT_TRUE(o.shape() != t.shape());
}

TEST(TensorClass, AsType) {
  Tensor t;
  EXPECT_EQ(singa::kFloat32, t.data_type());
  t.AsType(singa::kFloat16);
  EXPECT_EQ(singa::kFloat16, t.data_type());
}

TEST(TensorClass, ToDevice) {
  Tensor t(Shape{2,3});
  EXPECT_EQ(static_cast<Device*>(&singa::hostDeviceSingleton), t.device());
  singa::CppDevice *dev = new singa::CppDevice(0, 1);
  t.ToDevice(dev);
  EXPECT_NE(static_cast<Device*>(&singa::hostDeviceSingleton), t.device());
}

TEST(TensorClass, CopyDataFromHostPtr) {
  float data[] = {1.0f, 2.0f, 3.0f};
  Tensor t(Shape{3});
  t.CopyDataFromHostPtr(data, 3);
  const float* dptr = static_cast<const float*>(t.blob()->data());
  EXPECT_FLOAT_EQ(1.0f, dptr[0]);
  EXPECT_FLOAT_EQ(2.0f, dptr[1]);
  EXPECT_FLOAT_EQ(3.0f, dptr[2]);
}

TEST(TensorClass, CopyData) {
  float data[] = {1.0f, 2.0f, 3.0f};
  Tensor t(Shape{3});
  t.CopyDataFromHostPtr(data, 3);

  Tensor o(Shape{3});
  o.CopyData(t);
  const float* dptr = static_cast<const float*>(o.blob()->data());
  EXPECT_FLOAT_EQ(1.0f, dptr[0]);
  EXPECT_FLOAT_EQ(2.0f, dptr[1]);
  EXPECT_FLOAT_EQ(3.0f, dptr[2]);
}

TEST(TensorClass, Clone) {
  float data[] = {1.0f, 2.0f, 3.0f};
  Tensor t(Shape{3});
  t.CopyDataFromHostPtr(data, 3);

  Tensor o = t.Clone();
  const float* dptr = static_cast<const float*>(o.blob()->data());
  EXPECT_FLOAT_EQ(1.0f, dptr[0]);
  EXPECT_FLOAT_EQ(2.0f, dptr[1]);
  EXPECT_FLOAT_EQ(3.0f, dptr[2]);
}

TEST(TensorClass, T) {
  Tensor t(Shape{2,3});
  EXPECT_FALSE(t.transpose());
  Tensor o = t.T();
  EXPECT_EQ(true, o.transpose());
  EXPECT_EQ(t.blob(), o.blob());
  EXPECT_EQ(t.data_type(), o.data_type());
  EXPECT_EQ(t.shape()[0],  o.shape()[1]);
  EXPECT_EQ(t.shape()[1],  o.shape()[0]);
}



