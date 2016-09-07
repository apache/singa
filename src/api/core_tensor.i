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
%include "std_vector.i"
%include "std_string.i"
%include "std_shared_ptr.i"

/*
%include "carrays.i"
%array_class(float, floatArray);
%array_class(int, intArray);
%array_class(char, charArray);
%array_class(double, doubleArray);
*/

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
#endif //USE_PYTHON

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
           std::shared_ptr<singa::Device> dev, DataType dtype = kFloat32);
    Tensor(const Tensor &from);

    std::shared_ptr<singa::Device> device() const;
/*
    template <typename DType> const DType* data() const;
    %template(floatData) data<float>;
    %template(intData) data<int>;
    %template(charData) data<char>;
    %template(doubleData) data<double>;
    */

    template <typename SType> void GetValue(SType* value, const size_t num);
    %template(floatGetValue) GetValue<float>;
    %template(intGetValue) GetValue<int>;

    const DataType data_type() const;
    const std::vector<size_t> &shape() const;
    const size_t shape(size_t idx) const;
    size_t nDim() const;
    bool transpose() const;
    size_t Size() const;
    size_t MemSize() const;
    void Reshape(const std::vector<size_t> &shape);
    void ResetLike(const Tensor &t);
    void AsType(DataType type);
    void ToDevice(std::shared_ptr<singa::Device> dev);
    void ToHost();
    float L2() const;
    float L1() const;

    template <typename SType> void SetValue(const SType x);
    %template(floatSetValue) SetValue<float>;
    /* TODO(chonho-01) other types */
    // --- other types

    template <typename DType> void CopyDataFromHostPtr(const DType *src,
                                                       const size_t num,
                                                       const size_t offset = 0);
    %template(floatCopyDataFromHostPtr) CopyDataFromHostPtr<float>;
    %template(intCopyDataFromHostPtr) CopyDataFromHostPtr<int>;
    // --- other types

    void CopyData(const Tensor &other);
    Tensor Clone() const;
    Tensor T() const;

    /* python has no assignment operator
    Tensor &operator=(const Tensor &t); */
    Tensor &operator+=(const Tensor &t);
    Tensor &operator-=(const Tensor &t);
    Tensor &operator*=(const Tensor &t);
    Tensor &operator/=(const Tensor &t);


    template <typename DType> Tensor &operator+=(const DType x);
    %template(iAdd_f) operator+=<float>;
    // --- other types

    template <typename DType> Tensor &operator-=(DType x);
    %template(iSub_f) operator-=<float>;
    // --- other types

    template <typename DType> Tensor &operator*=(DType x);
    %template(iMul_f) operator*=<float>;
    // --- other types

    template <typename DType> Tensor &operator/=(DType x);
    %template(iDiv_f) operator/=<float>;
    // --- other types


    /*TODO(chonho-04)
    amax
    amin
    asum
    */


  };

  void CopyDataToFrom(Tensor *dst, const Tensor &src, size_t num,
                      size_t src_offset = 0, size_t dst_offset = 0);

  Tensor Reshape(const Tensor &in, const std::vector<size_t> &s);

  Tensor Abs(const Tensor &t);
  Tensor Exp(const Tensor &t);
  Tensor Log(const Tensor &t);
  Tensor ReLU(const Tensor &t);
  Tensor Sigmoid(const Tensor &t);
  Tensor Sign(const Tensor &t);
  Tensor Sqrt(const Tensor &t);
  Tensor Square(const Tensor &t);
  Tensor Tanh(const Tensor &t);

  Tensor Sum(const Tensor &t, int axis);
  template <typename SType> SType Sum(const Tensor &t);
  %template(floatSum) Sum<float>;
  // --- other types

  /* TODO(chonho-02)
     need to implement the average of all elements ??? */
  Tensor Average(const Tensor &t, int axis);
  Tensor SoftMax(const Tensor &t);


  Tensor Pow(const Tensor &base, const Tensor &exp);
  void Pow(const Tensor &base, const Tensor &exp, Tensor *out);

  %rename(Pow_f) Pow(const Tensor &in, const float x);
  template <typename SType>
  Tensor Pow(const Tensor &in, const SType x);
  %template(pow_temp) Pow<float>;

  %rename(Pow_f_out) Pow(const Tensor &in, const float x, Tensor *out);
  template <typename SType>
  void Pow(const Tensor &in, const SType x, Tensor *out);
  %template(pow_temp) Pow<float>;


  /* rename comparison operators */
  %rename(LT_Tf) operator<(const Tensor &t, const float x);
  %rename(LE_Tf) operator<=(const Tensor &t, const float x);
  %rename(GT_Tf) operator>(const Tensor &t, const float x);
  %rename(GE_Tf) operator>=(const Tensor &t, const float x);
  %rename(LT_TT) operator<(const Tensor &lhs, const Tensor &rhs);
  %rename(LE_TT) operator<=(const Tensor &lhs, const Tensor &rhs);
  %rename(GT_TT) operator>(const Tensor &lhs, const Tensor &rhs);
  %rename(GE_TT) operator>=(const Tensor &lhs, const Tensor &rhs);

  Tensor operator<(const Tensor &lhs, const Tensor &rhs);
  Tensor operator<=(const Tensor &lhs, const Tensor &rhs);
  Tensor operator>(const Tensor &lhs, const Tensor &rhs);
  Tensor operator>=(const Tensor &lhs, const Tensor &rhs);


  template <typename DType>
  Tensor operator<(const Tensor &t, const DType x);
  %template(op) operator< <float>;
  // --- other types

  template <typename DType>
  Tensor operator<=(const Tensor &t, const DType x);
  %template(op) operator<= <float>;
  // --- other types

  template <typename DType>
  Tensor operator>(const Tensor &t, const DType x);
  %template(op) operator> <float>;
  // --- other types

  template <typename DType>
  Tensor operator>=(const Tensor &t, const DType x);
  %template(op) operator>= <float>;
  // --- other types

  /* NOTE(chonho)
  no need to include theses
  in python, these can be replaced with comparison operators

  template <typename DType>
  void LT(const Tensor &t, DType x, Tensor *ret);
  template <typename DType>
  void LE(const Tensor &t, DType x, Tensor *ret);
  template <typename DType>
  void GT(const Tensor &t, DType x, Tensor *ret);
  template <typename DType>
  void GE(const Tensor &t, DType x, Tensor *ret);
  */


  /* ========== Arithmetic operations ========== */
  %rename(Add_TT) operator+(const Tensor &lhs, const Tensor &rhs);
  %rename(Sub_TT) operator-(const Tensor &lhs, const Tensor &rhs);
  %rename(EltwiseMul_TT) operator*(const Tensor &lhs, const Tensor &rhs);
  %rename(Div_TT) operator/(const Tensor &lhs, const Tensor &rhs);
  Tensor operator+(const Tensor &lhs, const Tensor &rhs);
  Tensor operator-(const Tensor &lhs, const Tensor &rhs);
  Tensor operator*(const Tensor &lhs, const Tensor &rhs);
  Tensor operator/(const Tensor &lhs, const Tensor &rhs);

  %rename(Add_Tf) operator+(const Tensor &t, float x);
  template <typename DType>
  Tensor operator+(const Tensor &t, DType x);
  %template(op) operator+<float>;
  // --- other types

  %rename(Sub_Tf) operator-(const Tensor &t, float x);
  template <typename DType>
  Tensor operator-(const Tensor &t, DType x);
  %template(op) operator-<float>;
  // --- other types

  %rename(EltwiseMul_Tf) operator*(const Tensor &t, float x);
  template <typename DType>
  Tensor operator*(const Tensor &t, DType x);
  %template(op) operator*<float>;
  // --- other types

  %rename(Div_Tf) operator/(const Tensor &t, float x);
  template <typename DType>
  Tensor operator/(const Tensor &t, DType x);
  %template(op) operator/<float>;
  // --- other types

  void Add(const Tensor &lhs, const Tensor &rhs, Tensor *ret);
  void Sub(const Tensor &lhs, const Tensor &rhs, Tensor *ret);
  void EltwiseMult(const Tensor &lhs, const Tensor &rhs, Tensor *ret);
  void Div(const Tensor &lhs, const Tensor &rhs, Tensor *ret);

  template <typename DType>
  void Add(const Tensor &t, DType x, Tensor *ret);
  %template(Add_Tf_out) Add<float>;
  // --- other types

  template <typename DType>
  void Sub(const Tensor &t, DType x, Tensor *ret);
  %template(Sub_Tf_out) Sub<float>;
  // --- other types

  template <typename DType>
  void EltwiseMult(const Tensor &t, DType x, Tensor *ret);
  %template(EltwiseMult_Tf_out) EltwiseMult<float>;
  // --- other types

  template <typename DType>
  void Div(const Tensor &t, DType x, Tensor *ret);
  %template(Div_Tf_out) Div<float>;
  // --- other types


  /* ========== Random operations ========== */
  template <typename SType>
  void Bernoulli(const SType p, Tensor *out);
  %template(floatBernoulli) Bernoulli<float>;
  // --- other types

  template <typename SType>
  void Gaussian(const SType mean, const SType std, Tensor *out);
  %template(floatGaussian) Gaussian<float>;
  // --- other types

  template <typename SType>
  void Uniform(const SType low, const SType high, Tensor *out);
  %template(floatUniform) Uniform<float>;
  // --- other types

  /* ========== Blas operations ========== */
  template <typename SType>
  void Axpy(SType alpha, const Tensor &in, Tensor *out);
  %template(floatAxpy) Axpy<float>;
  // --- other types

  Tensor Mult(const Tensor &A, const Tensor &B);
  void Mult(const Tensor &A, const Tensor &B, Tensor *C);
  template <typename SType>
  void Mult(const SType alpha, const Tensor &A, const Tensor &B,
            const SType beta, Tensor *C);
  %template(floatMult) Mult<float>;

  void AddColumn(const Tensor &v, Tensor *M);
  template <typename SType>
  void AddColumn(const SType alpha, const SType beta, const Tensor &v,
                 Tensor *M);
  %template(floatAddColumn) AddColumn<float>;

  void AddRow(const Tensor &v, Tensor *M);
  template <typename SType>
  void AddRow(const SType alpha, const SType beta, const Tensor &v,
              Tensor *M);
  %template(floatAddRow) AddRow<float>;

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

}

