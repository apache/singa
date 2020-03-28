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

/*interface file for swig */

%module core_tensor
%include "config.i"
%include "std_vector.i"
%include "std_string.i"
%include "std_shared_ptr.i"

%{
#define SWIG_FILE_WITH_INIT
#include "singa/core/tensor.h"
#include "singa/core/device.h"
#include "singa/proto/core.pb.h"
#include "singa/proto/model.pb.h"
using singa::DataType;
%}
%shared_ptr(singa::Device)

#if USE_PYTHON
%include "numpy.i"
%init %{
  import_array();
%}
%apply (float *IN_ARRAY1, int DIM1) {
       (const float *src, const size_t num)
}
%apply (int *IN_ARRAY1, int DIM1) {
       (const int *src, const size_t num)
}
%apply (float *ARGOUT_ARRAY1, int DIM1) {
       (float *value, const size_t num)
}
%apply (int *ARGOUT_ARRAY1, int DIM1) {
       (int *value, const size_t num)
}
#endif // USE_PYTHON

#if USE_JAVA
%include "arrays_java.i"
%apply int[] {int *};
%apply float[] {float *};
#endif // USE_JAVA



%template(Shape) std::vector<size_t>;

namespace singa{

  enum DataType {
    kFloat32, kFloat16, kInt, kChar, kDouble
  };

  inline size_t Product(const std::vector<size_t> &shape,
                        int start = 0, size_t len = 0);
  inline size_t SizeOf(DataType t);


  class Tensor {

   public:
    Tensor();
    explicit Tensor(const std::vector<size_t> &shape,
                    DataType dtype = kFloat32);
    Tensor(const std::vector<size_t> &shape,
           std::shared_ptr<singa::Device> dev,
           DataType dtype = kFloat32);
    Tensor(const Tensor &from);

    std::shared_ptr<singa::Device> device() const;

    template <typename SType> void GetValue(SType* value, const size_t num);
    %template(GetFloatValue) GetValue<float>;
    %template(GetIntValue) GetValue<int>;

    template <typename SType> void SetValue(const SType x);
    %template(SetFloatValue) SetValue<float>;

    const DataType data_type() const;
    const std::vector<size_t> &shape() const;
    const size_t shape(size_t idx) const;
    bool transpose() const;
    size_t nDim() const;    

    size_t Size() const;
    size_t MemSize() const;
    
    void ResetLike(const Tensor &t);
    Tensor AsType(DataType type);
    void ToDevice(std::shared_ptr<singa::Device> dev);
    void ToHost();
    float L2() const;
    float L1() const;

    template <typename DType> void CopyDataFromHostPtr(const DType *src,
                                                       const size_t num,
                                                       const size_t offset = 0);
    %template(CopyFloatDataFromHostPtr) CopyDataFromHostPtr<float>;
    %template(CopyIntDataFromHostPtr) CopyDataFromHostPtr<int>;

    void CopyData(const Tensor &other);
    void RepeatData(std::vector<size_t> repeats, int axis, int total_repeats, const Tensor &src);
    
    Tensor Clone() const;
    Tensor Repeat(std::vector<size_t> repeats, int axis);
    

#if USE_JAVA
    %rename(iAdd) operator+=(const Tensor &t);
    %rename(iSub) operator-=(const Tensor &t);
    %rename(iMul) operator*=(const Tensor &t);
    %rename(iDiv) operator/=(const Tensor &t);
#endif  // USE_JAVA

    Tensor &operator+=(const Tensor &t);
    Tensor &operator-=(const Tensor &t);
    Tensor &operator*=(const Tensor &t);
    Tensor &operator/=(const Tensor &t);

    template <typename DType> Tensor &operator+=(const DType x);
    %template(iAddFloat) operator+=<float>;

    template <typename DType> Tensor &operator-=(DType x);
    %template(iSubFloat) operator-=<float>;

    template <typename DType> Tensor &operator*=(DType x);
    %template(iMulFloat) operator*=<float>;

    template <typename DType> Tensor &operator/=(DType x);
    %template(iDivFloat) operator/=<float>;


    /*TODO(chonho-04)
    amax
    amin
    asum
    */
  };

  void CopyDataToFrom(Tensor *dst, const Tensor &src, size_t num,
                      size_t src_offset = 0, size_t dst_offset = 0);

  void RepeatDataToFrom(bool broadcast_flag, std::vector<size_t> repeats, int axis, 
                        Tensor *dst, const Tensor &src, const size_t num);

  Tensor Reshape(const Tensor &in, const std::vector<size_t> &s);
  Tensor Transpose(const Tensor &in, const std::vector<size_t> &axes);

  %rename(DefaultTranspose) Transpose(const Tensor &in);
  Tensor Transpose(const Tensor &in);

  Tensor Abs(const Tensor &t);
  Tensor Ceil(const Tensor &t);
  Tensor Exp(const Tensor &t);
  Tensor Log(const Tensor &t);
  Tensor ReLU(const Tensor &t);
  Tensor Sigmoid(const Tensor &t);
  Tensor Sign(const Tensor &t);
  Tensor Sqrt(const Tensor &t);
  Tensor Square(const Tensor &t);
  Tensor Cos(const Tensor &t);
  Tensor Cosh(const Tensor &t);
  Tensor Acos(const Tensor &t);
  Tensor Acosh(const Tensor &t);
  Tensor Sin(const Tensor &t);
  Tensor Sinh(const Tensor &t);
  Tensor Asin(const Tensor &t);
  Tensor Asinh(const Tensor &t);
  Tensor Tan(const Tensor &t);
  Tensor Tanh(const Tensor &t);
  Tensor Atan(const Tensor &t);
  Tensor Atanh(const Tensor &t);

  Tensor ReLUBackward(const Tensor &in1, const Tensor& in2);

  Tensor Sum(const Tensor &t, int axis);
  template <typename SType> SType Sum(const Tensor &t);
  %template(SumAsFloat) Sum<float>;
  Tensor SumAll(const Tensor &t);

  Tensor Average(const Tensor &t, int axis);
  Tensor SoftMax(const Tensor &t);
  Tensor SoftMax(const Tensor &t, int axis);
  Tensor SoftMaxBackward(const Tensor &t, int axis, const Tensor &fdout);

  Tensor Pow(const Tensor &base, const Tensor &exp);

  %rename(PowWithRet) Pow(const Tensor &base, const Tensor &exp, Tensor *out);
  void Pow(const Tensor &base, const Tensor &exp, Tensor *out);

  template <typename SType> Tensor Pow(const Tensor &in, const SType x);
  %template(PowFloat) Pow<float>;

  template <typename SType>
  void Pow(const Tensor &in, const SType x, Tensor *out);
  %template(PowFloatWithRet) Pow<float>;


  %rename(__lt__) operator<(const Tensor &lhs, const Tensor &rhs);
  %rename(__le__) operator<=(const Tensor &lhs, const Tensor &rhs);
  %rename(__gt__) operator>(const Tensor &lhs, const Tensor &rhs);
  %rename(__ge__) operator>=(const Tensor &lhs, const Tensor &rhs);
  Tensor operator<(const Tensor &lhs, const Tensor &rhs);
  Tensor operator<=(const Tensor &lhs, const Tensor &rhs);
  Tensor operator>(const Tensor &lhs, const Tensor &rhs);
  Tensor operator>=(const Tensor &lhs, const Tensor &rhs);


  %rename(LTFloat) operator<(const Tensor &t, const float x);
  template <typename DType>
  Tensor operator<(const Tensor &t, const DType x);
  %template(oplt) operator< <float>;

  %rename(LEFloat) operator<=(const Tensor &t, const float x);
  template <typename DType> Tensor operator<=(const Tensor &t, const DType x);
  %template(ople) operator<= <float>;

  %rename(GTFloat) operator>(const Tensor &t, const float x);
  template <typename DType> Tensor operator>(const Tensor &t, const DType x);
  %template(opgt) operator> <float>;

  %rename(GEFloat) operator>=(const Tensor &t, const float x);
  template <typename DType> Tensor operator>=(const Tensor &t, const DType x);
  %template(opge) operator>= <float>;

  Tensor ConcatOn(const std::vector<Tensor> &in, int axis);
  Tensor SliceOn(const Tensor&in, const size_t start, const size_t end, int axis);


  /* ========== Arithmetic operations ========== */
  %rename(__add__) operator+(const Tensor &lhs, const Tensor &rhs);
  %rename(__sub__) operator-(const Tensor &lhs, const Tensor &rhs);
  %rename(__mul__) operator*(const Tensor &lhs, const Tensor &rhs);
  %rename(__div__) operator/(const Tensor &lhs, const Tensor &rhs);
  Tensor operator+(const Tensor &lhs, const Tensor &rhs);
  Tensor operator-(const Tensor &lhs, const Tensor &rhs);
  Tensor operator*(const Tensor &lhs, const Tensor &rhs);
  Tensor operator/(const Tensor &lhs, const Tensor &rhs);
  void Add(const Tensor &lhs, const Tensor &rhs, Tensor *ret);
  void Sub(const Tensor &lhs, const Tensor &rhs, Tensor *ret);
  void EltwiseMult(const Tensor &lhs, const Tensor &rhs, Tensor *ret);
  void Div(const Tensor &lhs, const Tensor &rhs, Tensor *ret);

  %rename(AddFloat) operator+(const Tensor &t, float x);
  template <typename DType> Tensor operator+(const Tensor &t, DType x);
  %template(opadd) operator+ <float>;

  %rename(SubFloat) operator-(const Tensor &t, float x);
  template <typename DType> Tensor operator-(const Tensor &t, DType x);
  %template(opsub) operator- <float>;

  %rename(MultFloat) operator*(const Tensor &t, float x);
  template <typename DType> Tensor operator*(const Tensor &t, DType x);
  %template(opmul) operator* <float>;

  %rename(DivFloat) operator/(const Tensor &t, float x);
  template <typename DType> Tensor operator/(const Tensor &t, DType x);
  %template(opdiv) operator/ <float>;

  template <typename DType> void Add(const Tensor &t, DType x, Tensor *ret);
  %template(AddFloatWithRet) Add<float>;

  template <typename DType>
  void Sub(const Tensor &t, DType x, Tensor *ret);
  %template(SubFloatWithRet) Sub<float>;

  template <typename DType>
  void EltwiseMult(const Tensor &t, DType x, Tensor *ret);
  %template(EltwiseMultFloatWithRet) EltwiseMult<float>;

  template <typename DType>
  void Div(const Tensor &t, DType x, Tensor *ret);
  %template(DivFloatWithRet) Div<float>;


  /* ========== Random operations ========== */
  template <typename SType>
  void Bernoulli(const SType p, Tensor *out);
  %template(Bernoulli) Bernoulli<float>;

  template <typename SType>
  void Gaussian(const SType mean, const SType std, Tensor *out);
  %template(Gaussian) Gaussian<float>;

  template <typename SType>
  void Uniform(const SType low, const SType high, Tensor *out);
  %template(Uniform) Uniform<float>;


  /* ========== Blas operations ========== */
  template <typename SType>
  void Axpy(SType alpha, const Tensor &in, Tensor *out);
  %template(Axpy) Axpy<float>;

  Tensor Mult(const Tensor &A, const Tensor &B);
  %rename(MultWithRet) Mult(const Tensor &A, const Tensor &B, Tensor *C);
  void Mult(const Tensor &A, const Tensor &B, Tensor *C);
  template <typename SType>
  void Mult(const SType alpha, const Tensor &A, const Tensor &B,
            const SType beta, Tensor *C);
  %template(MultWithScale) Mult<float>;


  /* =========== Matrix operations ==========*/

  void AddColumn(const Tensor &v, Tensor *M);
  template <typename SType>
  void AddColumn(const SType alpha, const SType beta, const Tensor &v,
                 Tensor *M);
  %template(AddColumnWithScale) AddColumn<float>;

  void AddRow(const Tensor &v, Tensor *M);
  template <typename SType>
  void AddRow(const SType alpha, const SType beta, const Tensor &v,
              Tensor *M);
  %template(AddRowWithScale) AddRow<float>;

  void DivColumn(const Tensor &v, Tensor *M);
  void DivRow(const Tensor &v, Tensor *M);
  void MultColumn(const Tensor &v, Tensor *M);
  void MultRow(const Tensor &v, Tensor *M);
  void SubColumn(const Tensor &v, Tensor *M);
  void SubRow(const Tensor &v, Tensor *M);

  void SumColumns(const Tensor &M, Tensor *v);
  void SumRows(const Tensor &M, Tensor *v);

  Tensor SoftMax(const Tensor &in);
  void SoftMax(const Tensor &in, Tensor *out);
  Tensor SoftMax(const Tensor &in, int axis);
  void SoftMax(const Tensor &in, Tensor *out, int axis);

  Tensor CrossEntropyFwd(const Tensor& p, const Tensor& t);
  Tensor SoftmaxCrossEntropyBwd(const Tensor& p, const Tensor& t);
}
