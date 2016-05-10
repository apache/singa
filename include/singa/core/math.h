/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef SINGA_CORE_MATH_H_
#define SINGA_CORE_MATH_H_
#include <type_traits>
#include "singa/core/common.h"
#include "singa/utils/logging.h"

namespace singa {

/// \file math.h Math functions for linear algebra, neural net and random
/// operations.
/// All functions have a template argument, DType for DataType, Lib for the
/// backend library, e.g., lib::Cublas, lib::Cudnn, etc.

/// Some operations would have many config/hyper-parameters, e.g., Conv, and
/// these config vary among diff implementations, e.g., cuda/cudnn/opencl.
/// To separate the modules, we pass a OpConf pointer to the Tensor Op function.
/// The specific fields are implemented by inheriting OpConf, and casting the
/// pointer between the base and the sub-class.
class OpConf {
 public:
  template <typename T>
  T* CastTo() {
    static_assert(std::is_base_of<OpConf, T>::value,
                  "The cast type must be a sub-class of OpConf");
    return static_cast<T*>(this);
  }
};

template <typename DType, typename Lib>
void Add(int count, const Blob* lhs, const Blob* rhs, Blob* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}

// ================Neural Net operations======================================

class ConvConf : public OpConf {};
template <typename DType, typename Lib>
void Conv(const OpConf* conf, const Blob* input, const Blob* W, const Blob* b,
          Blob* ret, Context* ctx) {
  LOG(FATAL) << "Not Implemented";
}
}  // namespace singa

#endif  // SINGA_CORE_MATH_H_
