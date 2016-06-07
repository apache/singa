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
#include "core/tensor/tensor_math.h"
#include "singa/core/tensor.h"
#include "singa/core/device.h"
#include "singa/proto/core.pb.h"
#include "singa/proto/model.pb.h"
using singa::Device;
using singa::DataType;
%}

%template(Shape) std::vector<size_t>;

namespace singa{

  //%rename(floatLT) operator<(const Tensor &t, const float x);


  enum DataType {
    kFloat32, kFloat16, kInt, kChar, kDouble
  };

  inline size_t Product(const std::vector<size_t> &shape,
                        int start = 0, size_t len = 0);
  inline size_t SizeOf(DataType t);

  class Tensor {

    /* renamed functions */
    /* TODO(chonho-02) assign check python code */
    %rename(Assign_T) operator=(const Tensor &t);


   public:
    Tensor();
    explicit Tensor(const std::vector<size_t> &shape, DataType dtype = kFloat32);
    Tensor(const std::vector<size_t> &shape, Device *dev, DataType dtype = kFloat32);
    Tensor(const Tensor &from);

    //Blob *blob() const;
    Device *device() const;

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
    void ToDevice(Device *dev);
    void ToHost();

    template <typename SType> void SetValue(const SType x);
    %template(floatSetValue) SetValue<float>;
    // ...

    template <typename DType> void CopyDataFromHostPtr(const DType *src, size_t num);
    %template(floatCopyDataFromHostPtr) CopyDataFromHostPtr<float>;
    // ...

    void CopyData(const Tensor &other);
    Tensor Clone() const;
    Tensor T() const;

    /* the following operators are renamed */
    /* TODO(chonho-02) assign rename check python code*/
    Tensor &operator=(const Tensor &t);

    Tensor &operator+=(const Tensor &t);
    Tensor &operator-=(const Tensor &t);
    Tensor &operator*=(const Tensor &t);
    Tensor &operator/=(const Tensor &t);

    template <typename DType> Tensor &operator+=(const DType x);
    %template(iAdd_f) operator+=<float>;
    /* TODO(chonho-01) for other types */
    //%template(iAdd_i) operator+=<int>;
    //%template(iAdd_c) operator+=<char>;
    //%template(iAdd_d) operator+=<double>;

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

  inline void CheckDataTypeAndLang(const Tensor &in1, const Tensor &in2);
  Tensor Reshape(const Tensor &in, const std::vector<size_t> &s);
  void CopyDataToFrom(Tensor *dst, const Tensor &src, size_t num,
                      size_t src_offset = 0, size_t dst_offset = 0);
  Tensor Abs(const Tensor &t);
  Tensor Exp(const Tensor &t);
  Tensor Log(const Tensor &t);
  Tensor ReLU(const Tensor &t);
  Tensor Sigmoid(const Tensor &t);
  Tensor Sign(const Tensor &t);
  Tensor Sqrt(const Tensor &t);
  Tensor Square(const Tensor &t);
  Tensor Tanh(const Tensor &t);

  template <typename SType> SType Sum(const Tensor &t);
  %template(floatSum) Sum<float>;
  /* TODO(chonho-03) not implemented
  %template(intSum) Sum<int>;
  %template(charSum) Sum<char>;
  %template(doubleSum) Sum<double>;
  */
  Tensor Sum(const Tensor &t, int axis);
  /* TODO(chonho-04) not implemented
     all average ??? */
  Tensor Average(const Tensor &t, int axis);
  Tensor SoftMax(const Tensor &t, int axis = 0);
  /* TODO(chonho-05) no need for python???
  void SoftMax(const Tensor &t, int axis, Tensor *ret);
  */




  /* TODO(chonho-06) operators  */
  /*
  template <typename DType>
  Tensor operator<(const Tensor &t, const DType x);
  %template(floatLT_op) operator< <float>;
  */
  //Tensor operator<(const Tensor &t, const float x);
  //Tensor operator<(const Tensor &t, const int x);
  //Tensor operator<(const Tensor &t, const char x);
  //Tensor operator<(const Tensor &t, const double x);

  /*
  template <typename DType>
  void LT(const Tensor &t, DType x, Tensor *ret);
  %template(floatLT) LT<float>;
  %template(intLT) LT<int>;
  %template(charLT) LT<char>;
  %template(doubleLT) LT<double>;
  */

  /* renamed
  template <typename DType>
  Tensor operator<=(const Tensor &t, const DType x); */
  // ---

  template <typename DType>
  void LE(const Tensor &t, DType x, Tensor *ret);
  %template(floatLE) LE<float>;
  // ---

  template <typename DType>
  Tensor operator>(const Tensor &t, const DType x);
  %template(floatGT_op) operator><float>;
  //Tensor operator>(const Tensor &t, const float x);
  // ---

  template <typename DType>
  void GT(const Tensor &t, DType x, Tensor *ret);
  %template(floatGT) GT<float>;
  // ---

  /* renamed
  template <typename DType>
  Tensor operator>=(const Tensor &t, const DType x); */
  Tensor operator>=(const Tensor &t, const float x);
  // ---

  template <typename DType>
  void GE(const Tensor &t, DType x, Tensor *ret);
  %template(floatGE) GE<float>;
  // ---

  /*
  template <typename DType>
  Tensor Pow(const Tensor &t, DType x);
  template <typename DType>
  void Pow(const Tensor &t, DType x, Tensor *ret);
  */
  //Tensor Pow(const Tensor &base, Tensor exp);
  //void Pow(const Tensor &base, const Tensor &exp, Tensor *ret);

  /* rename operators */
  %rename(Add_TT) operator+(const Tensor &lhs, const Tensor &rhs);
  %rename(Sub_TT) operator-(const Tensor &lhs, const Tensor &rhs);
  %rename(Mul_TT) operator*(const Tensor &lhs, const Tensor &rhs);
  %rename(Div_TT) operator/(const Tensor &lhs, const Tensor &rhs);
  Tensor operator+(const Tensor &lhs, const Tensor &rhs);
  Tensor operator-(const Tensor &lhs, const Tensor &rhs);
  Tensor operator*(const Tensor &lhs, const Tensor &rhs);
  Tensor operator/(const Tensor &lhs, const Tensor &rhs);

  void Add(const Tensor &lhs, const Tensor &rhs, Tensor *ret);
  void Sub(const Tensor &lhs, const Tensor &rhs, Tensor *ret);
  void EltwiseMult(const Tensor &lhs, const Tensor &rhs, Tensor *ret);
  void Div(const Tensor &lhs, const Tensor &rhs, Tensor *ret);

  /* (TODO) other operators */
}

