#include "gtest/gtest.h"
#include "singa/core/tensor.h"
using singa::Tensor;
using singa::Shape;
using singa::Device;

class TestTensorMath : public ::testing::Test {
protected:
  virtual void SetUp() {
    a.Reshape(singa::Shape{6});
    b.Reshape(singa::Shape{6});
    c.Reshape(singa::Shape{6, 1});
    d.Reshape(singa::Shape{3, 2});

    a.CopyDataFromHostPtr<float>(dat1, 6);
    b.CopyDataFromHostPtr<float>(dat2, 6);
  }
  Tensor a, b, c, d;
  const float dat1[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  const float dat2[6] = {1.1f, 2.1f, 3.1f, 4.1f, 5.1f, 6.1f};
};

TEST_F(TestTensorMath, MemberAddTensor) {
  Tensor aa = a.Clone();
  aa += a;
  const float *dptr = aa.data<const float *>();
  EXPECT_FLOAT_EQ(2.0f, dptr[0]);
  EXPECT_FLOAT_EQ(4.0f, dptr[1]);
  EXPECT_FLOAT_EQ(6.0f, dptr[2]);

  // check p is initialized to 0
  Tensor p(Shape{6});
  p += aa;
  const float *dptr1 = p.data<const float *>();
  EXPECT_FLOAT_EQ(2.0f, dptr1[0]);
  EXPECT_FLOAT_EQ(4.0f, dptr1[1]);
  EXPECT_FLOAT_EQ(6.0f, dptr1[2]);

  a += b;
  const float *dptr2 = a.data<const float *>();
  EXPECT_FLOAT_EQ(2.1f, dptr2[0]);
  EXPECT_FLOAT_EQ(4.1f, dptr2[1]);
  EXPECT_FLOAT_EQ(6.1f, dptr2[2]);
  EXPECT_FLOAT_EQ(12.1f, dptr2[5]);
}

TEST_F(TestTensorMath, AddTensors) {
  Tensor ret(a.shape(), a.device(), a.data_type());
  Add(a, b, &ret);
  const float *dptr = ret.data<const float *>();
  EXPECT_FLOAT_EQ(2.1f, dptr[0]);
  EXPECT_FLOAT_EQ(4.1f, dptr[1]);
  EXPECT_FLOAT_EQ(6.1f, dptr[2]);
  EXPECT_FLOAT_EQ(12.1f, dptr[5]);

  const Tensor d = a + b;
  const float *dptr2 = d.data<const float *>();
  EXPECT_FLOAT_EQ(2.1f, dptr2[0]);
  EXPECT_FLOAT_EQ(4.1f, dptr2[1]);
  EXPECT_FLOAT_EQ(6.1f, dptr2[2]);
  EXPECT_FLOAT_EQ(12.1f, dptr2[5]);

  Add(a, b, &a);
  const float *dptr1 = a.data<const float *>();
  EXPECT_FLOAT_EQ(2.1f, dptr1[0]);
  EXPECT_FLOAT_EQ(4.1f, dptr1[1]);
  EXPECT_FLOAT_EQ(6.1f, dptr1[2]);
  EXPECT_FLOAT_EQ(12.1f, dptr1[5]);
}

TEST_F(TestTensorMath, SetValue) {
  Tensor t(Shape{4});
  t.SetValue(0.3f);
  const float *ptr = t.data<const float *>();
  for (int i = 0; i < 4; i++)
    EXPECT_FLOAT_EQ(ptr[i], 0.3f);
}

TEST_F(TestTensorMath, Reshape) {
  Tensor t(Shape{4});
  t.SetValue(0.3f);
  Tensor p = Reshape(t, Shape{4, 1});
  const float *ptr = t.data<const float *>();
  EXPECT_EQ(p.shape(0), 4u);
  EXPECT_EQ(p.shape(1), 1u);
  for (int i = 0; i < 4; i++)
    EXPECT_FLOAT_EQ(ptr[i], 0.3f);
}
#ifdef USE_CBLAS
TEST_F(TestTensorMath, MultCpp) {
  const float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  Tensor t(Shape{2, 2});
  t.CopyDataFromHostPtr(x, 4);
  d.CopyDataFromHostPtr(dat1, 6);
  Tensor C = Mult(d, t);
  const float *xptr = C.data<const float *>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      float tmp = 0;
      for (int k = 0; k < 2; k++) {
        tmp += dat1[i * 2 + k] * x[k * 2 + j];
      }
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], tmp);
    }
  }
  const float y[8] = {1.0f, 2.0f, 3.0f, 4.0f, 1.1f, 2.1f, 3.1f, 4.1f};
  Tensor s(Shape{4, 2});
  s.CopyDataFromHostPtr(y, 8);
  const float *sPtr = s.data<const float *>();
  for (int i = 0; i < 8; i++)
    EXPECT_FLOAT_EQ(sPtr[i], y[i]);
  Tensor D = Mult(d, s.T());
  const float *DPtr = D.data<const float *>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      float tmp = 0;
      for (int k = 0; k < 2; k++) {
        tmp += dat1[i * 2 + k] * y[j * 2 + k];
      }
      EXPECT_FLOAT_EQ(DPtr[i * 4 + j], tmp);
    }
  }
  Tensor p(Shape{4, 1});
  p.CopyDataFromHostPtr(x, 4);
  Tensor q(Shape{1, 4});
  q.SetValue(1.0f);
  Tensor o(Shape{4, 4});

  Mult(p, q, &o);
  const float *oPtr = o.data<const float *>();
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      EXPECT_FLOAT_EQ(oPtr[i * 4 + j], x[i]);
    }
  }
}

TEST_F(TestTensorMath, AddColumnCpp) {
  const float x[3] = {1.0f, 2.0f, 3.0f};
  Tensor t(Shape{3});
  t.CopyDataFromHostPtr(x, 3);
  d.CopyDataFromHostPtr(dat1, 6);
  AddColumn(t, &d);
  const float *xptr = d.data<const float *>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] + x[i]);
    }
  }
}
TEST_F(TestTensorMath, SubColumnCpp) {
  const float x[3] = {1.0f, 2.0f, 3.0f};
  Tensor t(Shape{3});
  t.CopyDataFromHostPtr(x, 3);
  d.CopyDataFromHostPtr(dat1, 6);
  SubColumn(t, &d);
  const float *xptr = d.data<const float *>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] - x[i]);
    }
  }
}


TEST_F(TestTensorMath, DivColumnCpp) {
  const float x[3] = {1.0f, 2.0f, 3.0f};
  Tensor t(Shape{3});
  t.CopyDataFromHostPtr(x, 3);
  d.CopyDataFromHostPtr(dat1, 6);
  DivColumn(t, &d);
  const float *xptr = d.data<const float *>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] / x[i]);
    }
  }
}


TEST_F(TestTensorMath, AddRowCpp) {
  const float x[2] = {1.1f, 2.1f};
  Tensor t(Shape{2});
  t.CopyDataFromHostPtr(x, 2);
  d.CopyDataFromHostPtr(dat1, 6);
  AddRow(t, &d);
  const float *xptr = d.data<const float *>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] + x[j]);
    }
  }
}


TEST_F(TestTensorMath, SubRowCpp) {
  const float x[2] = {1.1f, 2.1f};
  Tensor t(Shape{2});
  t.CopyDataFromHostPtr(x, 2);
  d.CopyDataFromHostPtr(dat1, 6);
  SubRow(t, &d);
  const float *xptr = d.data<const float *>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] - x[j]);
    }
  }
}


TEST_F(TestTensorMath, MultRowCpp) {
  const float x[2] = {1.1f, 2.1f};
  Tensor t(Shape{2});
  t.CopyDataFromHostPtr(x, 2);
  d.CopyDataFromHostPtr(dat1, 6);
  MultRow(t, &d);
  const float *xptr = d.data<const float *>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] * x[j]);
    }
  }
}


TEST_F(TestTensorMath, SumRowsCpp) {
  Tensor t(Shape{2});
  d.CopyDataFromHostPtr(dat1, 6);
  SumRows(d, &t);
  const float *tptr = t.data<const float *>();
  for (int i = 0; i < 2; i++) {
    float tmp = 0;
    for (int j = 0; j < 3; j++) {
      tmp += dat1[j * 2 + i];
    }
    EXPECT_FLOAT_EQ(tptr[i], tmp);
  }
}


TEST_F(TestTensorMath, SumColumnsCpp) {
  Tensor t(Shape{3});
  d.CopyDataFromHostPtr(dat1, 6);
  SumColumns(d, &t);
  const float *tptr = t.data<const float *>();
  for (int i = 0; i < 3; i++) {
    float tmp = 0;
    for (int j = 0; j < 2; j++) {
      tmp += dat1[i * 2 + j];
    }
    EXPECT_FLOAT_EQ(tptr[i], tmp);
  }
}
#endif
#ifdef USE_CUDA
TEST_F(TestTensorMath, MultCuda) {
  const float x[4] = {1.0f, 2.0f, 3.0f, 4.0f};
  singa::CudaGPU dev;
  Tensor t(Shape{2, 2}, &dev);
  t.CopyDataFromHostPtr(x, 4);
  d.ToDevice(&dev);
  d.CopyDataFromHostPtr(dat1, 6);
  Tensor C = Mult(d, t);
  C.ToHost();
  const float *xptr = C.data<const float *>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      float tmp = 0;
      for (int k = 0; k < 2; k++) {
        tmp += dat1[i * 2 + k] * x[k * 2 + j];
      }
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], tmp);
    }
  }

  const float y[8] = {1.0f, 2.0f, 3.0f, 4.0f, 1.1f, 2.1f, 3.1f, 4.1f};
  Tensor s(Shape{4, 2}, &dev);
  s.CopyDataFromHostPtr(y, 8);
  Tensor D = Mult(d, s.T());
  D.ToHost();
  const float *DPtr = D.data<const float *>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
      float tmp = 0;
      for (int k = 0; k < 2; k++) {
        tmp += dat1[i * 2 + k] * y[j * 2 + k];
      }
      EXPECT_FLOAT_EQ(DPtr[i * 4 + j], tmp);
    }
  }
  Tensor p(Shape{4, 1}, &dev);
  p.CopyDataFromHostPtr(x, 4);
  Tensor q(Shape{1, 4}, &dev);
  q.SetValue(1.0f);
  Tensor o(Shape{4, 4}, &dev);

  Mult(p, q, &o);
  o.ToHost();
  const float *oPtr = o.data<const float *>();
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      EXPECT_FLOAT_EQ(oPtr[i * 4 + j], x[i]);
    }
  }
}

TEST_F(TestTensorMath, AddColumnCuda) {
  const float x[3] = {1.0f, 2.0f, 3.0f};
  singa::CudaGPU dev;
  Tensor t(Shape{3}, &dev);
  t.CopyDataFromHostPtr(x, 3);
  d.CopyDataFromHostPtr(dat1, 6);
  d.ToDevice(&dev);
  AddColumn(t, &d);
  d.ToHost();
  const float *xptr = d.data<const float *>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] + x[i]);
    }
  }
}


TEST_F(TestTensorMath, SubColumnCuda) {
  const float x[3] = {1.0f, 2.0f, 3.0f};
  singa::CudaGPU dev;
  Tensor t(Shape{3}, &dev);
  t.CopyDataFromHostPtr(x, 3);
  d.CopyDataFromHostPtr(dat1, 6);
  d.ToDevice(&dev);
  SubColumn(t, &d);
  d.ToHost();
  const float *xptr = d.data<const float *>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] - x[i]);
    }
  }
}
#endif
TEST_F(TestTensorMath, MultColumnCpp) {
  const float x[3] = {1.0f, 2.0f, 3.0f};
  Tensor t(Shape{3});
  t.CopyDataFromHostPtr(x, 3);
  d.CopyDataFromHostPtr(dat1, 6);
  MultColumn(t, &d);
  const float *xptr = d.data<const float *>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] * x[i]);
    }
  }
}
#ifdef USE_CUDA
TEST_F(TestTensorMath, MultColumnCuda) {
  const float x[3] = {1.0f, 2.0f, 3.0f};
  singa::CudaGPU dev;
  Tensor t(Shape{3}, &dev);
  t.CopyDataFromHostPtr(x, 3);
  d.CopyDataFromHostPtr(dat1, 6);
  d.ToDevice(&dev);
  MultColumn(t, &d);
  d.ToHost();
  const float *xptr = d.data<const float *>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] * x[i]);
    }
  }
}
TEST_F(TestTensorMath, DivColumnCuda) {
  const float x[3] = {1.0f, 2.0f, 3.0f};
  singa::CudaGPU dev;
  Tensor t(Shape{3}, &dev);
  t.CopyDataFromHostPtr(x, 3);
  d.CopyDataFromHostPtr(dat1, 6);
  d.ToDevice(&dev);
  DivColumn(t, &d);
  d.ToHost();
  const float *xptr = d.data<const float *>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] / x[i]);
    }
  }
}
TEST_F(TestTensorMath, AddRowCuda) {
  const float x[2] = {1.1f, 2.1f};
  singa::CudaGPU dev;
  Tensor t(Shape{2}, &dev);
  t.CopyDataFromHostPtr(x, 2);
  d.CopyDataFromHostPtr(dat1, 6);
  d.ToDevice(&dev);
  AddRow(t, &d);
  d.ToHost();
  const float *xptr = d.data<const float *>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] + x[j]);
    }
  }
}
TEST_F(TestTensorMath, SubRowCuda) {
  const float x[2] = {1.1f, 2.1f};
  singa::CudaGPU dev;
  Tensor t(Shape{2}, &dev);
  t.CopyDataFromHostPtr(x, 2);
  d.CopyDataFromHostPtr(dat1, 6);
  d.ToDevice(&dev);
  SubRow(t, &d);
  d.ToHost();
  const float *xptr = d.data<const float *>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] - x[j]);
    }
  }
}
TEST_F(TestTensorMath, MultRowCuda) {
  const float x[2] = {1.1f, 2.1f};
  singa::CudaGPU dev;
  Tensor t(Shape{2}, &dev);
  t.CopyDataFromHostPtr(x, 2);
  d.CopyDataFromHostPtr(dat1, 6);
  d.ToDevice(&dev);
  MultRow(t, &d);
  d.ToHost();
  const float *xptr = d.data<const float *>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] * x[j]);
    }
  }
}
#endif
TEST_F(TestTensorMath, DivRowCpp) {
  const float x[2] = {1.1f, 2.1f};
  Tensor t(Shape{2});
  t.CopyDataFromHostPtr(x, 2);
  d.CopyDataFromHostPtr(dat1, 6);
  DivRow(t, &d);
  const float *xptr = d.data<const float *>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] / x[j]);
    }
  }
}
#ifdef USE_CUDA
TEST_F(TestTensorMath, DivRowCuda) {
  const float x[2] = {1.1f, 2.1f};
  singa::CudaGPU dev;
  Tensor t(Shape{2}, &dev);
  t.CopyDataFromHostPtr(x, 2);
  d.CopyDataFromHostPtr(dat1, 6);
  d.ToDevice(&dev);
  DivRow(t, &d);
  d.ToHost();
  const float *xptr = d.data<const float *>();
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 2; j++) {
      EXPECT_FLOAT_EQ(xptr[i * 2 + j], dat1[i * 2 + j] / x[j]);
    }
  }
}
TEST_F(TestTensorMath, SumRowsCuda) {
  singa::CudaGPU dev;
  Tensor t(Shape{2}, &dev);
  d.CopyDataFromHostPtr(dat1, 6);
  d.ToDevice(&dev);
  SumRows(d, &t);
  t.ToHost();
  const float *tptr = t.data<const float *>();
  for (int i = 0; i < 2; i++) {
    float tmp = 0;
    for (int j = 0; j < 3; j++) {
      tmp += dat1[j * 2 + i];
    }
    EXPECT_FLOAT_EQ(tptr[i], tmp);
  }
}
TEST_F(TestTensorMath, SumColumnCuda) {
  singa::CudaGPU dev;
  Tensor t(Shape{3}, &dev);
  d.CopyDataFromHostPtr(dat1, 6);
  d.ToDevice(&dev);
  SumColumns(d, &t);
  t.ToHost();
  const float *tptr = t.data<const float *>();
  for (int i = 0; i < 3; i++) {
    float tmp = 0;
    for (int j = 0; j < 2; j++) {
      tmp += dat1[i * 2 + j];
    }
    EXPECT_FLOAT_EQ(tptr[i], tmp);
  }
}
#endif
