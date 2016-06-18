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

%include "carrays.i"
%array_class(float, floatArray);
%array_class(int, intArray);
%array_class(char, charArray);
%array_class(double, doubleArray);

%{
#include "singa/core/tensor.h"
#include "singa/core/device.h"
#include "singa/proto/core.pb.h"
#include "singa/proto/model.pb.h"
using singa::DataType;
%}

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
           singa::Device *dev, DataType dtype = kFloat32);
    Tensor(const Tensor &from);

    //Blob *blob() const;
    singa::Device *device() const;

    template <typename DType> DType data() const;
    %template(floatData) data<const float*>;
    %template(intData) data<const int*>;
    %template(charData) data<const char*>;
    %template(doubleData) data<const double*>;

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
    void ToDevice(singa::Device *dev);
    void ToHost();

    template <typename SType> void SetValue(const SType x);
    %template(floatSetValue) SetValue<float>;
    // ...

    /* no need to expose this function
    template <typename DType> void CopyDataFromHostPtr(const DType *src,
                                                       size_t num);
    */

    void CopyData(const Tensor &other);
    Tensor Clone() const;
    Tensor T() const;

    /* python has no assignment operator as c++
    Tensor &operator=(const Tensor &t); */
    Tensor &operator+=(const Tensor &t);
    Tensor &operator-=(const Tensor &t);
    Tensor &operator*=(const Tensor &t);
    Tensor &operator/=(const Tensor &t);


    template <typename DType> Tensor &operator+=(const DType x);
    %template(iAdd_f) operator+=<float>;
    /* TODO(chonho-01) for other types */
    // ...

    template <typename DType> Tensor &operator-=(DType x);
    %template(iSub_f) operator-=<float>;
    /* TODO(chonho-01) for other types */
    // ...

    template <typename DType> Tensor &operator*=(DType x);
    %template(iMul_f) operator*=<float>;
    /* TODO(chonho-01) for other types */
    // ...

    template <typename DType> Tensor &operator/=(DType x);
    %template(iDiv_f) operator/=<float>;
    /* TODO(chonho-01) for other types */
    // ...

  };

  /* TODO
  inline void CheckDataTypeAndLang(const Tensor &in1, const Tensor &in2);
  */
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
  /* TODO(chonho-03) not implemented
  %template(intSum) Sum<int>;
  %template(charSum) Sum<char>;
  %template(doubleSum) Sum<double>;
  */

  /* TODO(chonho-04) not implemented
     need average of all elements ??? */
  Tensor Average(const Tensor &t, int axis);
  Tensor SoftMax(const Tensor &t);

  /* TODO(chonho-05) not implemented ???
  Tensor Pow(const Tensor &base, Tensor exp);
  template <typename DType>
  Tensor Pow(const Tensor &t, DType x);
  */


  /* rename comparison operators */
  %rename(LT_Tf) operator<(const Tensor &t, const float x);
  %rename(LE_Tf) operator<=(const Tensor &t, const float x);
  %rename(GT_Tf) operator>(const Tensor &t, const float x);
  %rename(GE_Tf) operator>=(const Tensor &t, const float x);

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

  /* TODO(chonho-06)
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


  /* rename operators */
  %rename(Add_TT) operator+(const Tensor &lhs, const Tensor &rhs);
  %rename(Sub_TT) operator-(const Tensor &lhs, const Tensor &rhs);
  %rename(Mul_TT) operator*(const Tensor &lhs, const Tensor &rhs);
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

  %rename(Mul_Tf) operator*(const Tensor &t, float x);
  template <typename DType>
  Tensor operator*(const Tensor &t, DType x);
  %template(op) operator*<float>;
  // --- other types

  %rename(Div_Tf) operator/(const Tensor &t, float x);
  template <typename DType>
  Tensor operator/(const Tensor &t, DType x);
  %template(op) operator/<float>;
  // --- other types

  /* TODO(chonho-07)
  no need to include theses
  in python, these can be replaced with operators

  void Add(const Tensor &lhs, const Tensor &rhs, Tensor *ret);
  void Sub(const Tensor &lhs, const Tensor &rhs, Tensor *ret);
  void EltwiseMult(const Tensor &lhs, const Tensor &rhs, Tensor *ret);
  void Div(const Tensor &lhs, const Tensor &rhs, Tensor *ret);

  template <typename DType>
  void Add(const Tensor &t, DType x, Tensor *ret);
  template <typename DType>
  void Sub(const Tensor &t, DType x, Tensor *ret);
  template <typename DType>
  void EltwiseMult(const Tensor &t, DType x, Tensor *ret);
  template <typename DType>
  void Div(const Tensor &t, DType x, Tensor *ret);
  */

}

