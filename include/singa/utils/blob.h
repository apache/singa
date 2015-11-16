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

/**
 * The code is adapted from that of Caffe whose license is attached.
 *
 * COPYRIGHT
 * All contributions by the University of California:
 * Copyright (c) 2014, The Regents of the University of California (Regents)
 * All rights reserved.
 * All other contributions:
 * Copyright (c) 2014, the respective contributors
 * All rights reserved.
 * Caffe uses a shared copyright model: each contributor holds copyright over
 * their contributions to Caffe. The project versioning records all such
 * contribution and copyright details. If a contributor wants to further mark
 * their specific copyright on a particular contribution, they should indicate
 * their copyright solely in the commit message of the change when it is
 * committed.
 * LICENSE
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 * CONTRIBUTION AGREEMENT
 * By contributing to the BVLC/caffe repository through pull-request, comment,
 * or otherwise, the contributor releases their content to the
 * license and copyright terms herein.
 *
 */
#ifndef SINGA_UTILS_BLOB_H_
#define SINGA_UTILS_BLOB_H_

#include <glog/logging.h>
#include <memory>
#include <vector>
#include "singa/proto/common.pb.h"
#include "mshadow/tensor.h"
#include "mshadow/cxxnet_op.h"

namespace singa {

inline void MallocHost(void** ptr, size_t size) {
  *ptr = malloc(size);
}

inline void FreeHost(void* ptr) {
  free(ptr);
}

/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
class SyncedMemory {
 public:
  enum SyncedHead { UNINITIALIZED,
                    HEAD_AT_CPU,
                    HEAD_AT_GPU,
                    SYNCED };

  SyncedMemory() {}
  explicit SyncedMemory(size_t size) : size_(size) {}
  ~SyncedMemory();

  const void* cpu_data();
  const void* gpu_data();
  void* mutable_cpu_data();
  void* mutable_gpu_data();
  void set_cpu_data(void* data);
  inline SyncedHead head() { return head_; }
  inline size_t size() { return size_; }

 private:
  void to_cpu();
  void to_gpu();

  void* cpu_ptr_ = nullptr;
  void* gpu_ptr_ = nullptr;
  size_t size_ = 0;
  SyncedHead head_ = UNINITIALIZED;
  bool own_cpu_data_ = false;
};  // class SyncedMemory


template <typename Dtype>
class Blob {
 public:
  Blob() {}
  /**
   * Blob constructor with given shape.
   * @param shape specifies the size of each dimension, shape[0] is the highest
   * dimension, i.e., stride[0] = shape[1] * shape[2] * ...
   */
  explicit Blob(const std::vector<int>& shape) { Reshape(shape); }
  /**
   * Blob constructor with given shape.
   * @param[in] dim0 total num of elements.
   */
  explicit Blob(int dim0) { Reshape(dim0); }
  /**
   * Blob constructor with given shape.
   * @param[in] dim0 size of the highest dimension
   * @param[in] dim1 size of the second highest dimension
   */
  explicit Blob(int dim0, int dim1) { Reshape(dim0, dim1); }
  /**
   * Blob constructor with given shape.
   * @param[in] dim0 size of the highest dimension
   * @param[in] dim1
   * @param[in] dim2
   */
  explicit Blob(int dim0, int dim1, int dim2) { Reshape(dim0, dim1, dim2); }
  /**
   * Blob constructor with given shape.
   * @param[in] dim0 size of the highest dimension
   * @param[in] dim1
   * @param[in] dim2
   * @param[in] dim3
   */
  explicit Blob(int dim0, int dim1, int dim2, int dim3) {
    Reshape(dim0, dim1, dim2, dim3);
  }
  /**
   * Change the shape of the blob, re-allocate memory if Blob size() changes.
   *
   * @param[in] shape specifies the size of each dimension, shape[0] is the
   * highest * dimension, i.e., stride[0] = shape[1] * shape[2] * ...
   */
  void Reshape(const std::vector<int>& shape);
  /**
   * Helper for Reshape(const std::vector<int>& shape) with shape.size() = 1.
   *
   * @see Reshape(const std::vector<int>&).
   * @param[in] dim0 total num of elements.
   */
  void Reshape(int dim0) {
    Reshape(std::vector<int>{dim0});
  }
  /**
   * Helper for Reshape(const std::vector<int>& shape) with shape.size() = 2.
   *
   * @param dim0 the highest dimension size, i.e., dim0 = shape[0]. E.g., dim0
   * could the batchsize.
   * @param[in] dim1, dim1 = shape[1], e.g., dim1 could be the length of the
   * feature vector.
   */
  void Reshape(int dim0, int dim1) {
    Reshape(std::vector<int>{dim0, dim1});
  }
  /**
   * Helper for Reshape(const std::vector<int>& shape) with shape.size() = 3.
   *
   * @param[in] dim0, dim0 = shape[0]
   * @param[in] dim1, dim1 = shape[1]
   * @param[in] dim2, dim2 = shape[2]
   */
  void Reshape(int dim0, int dim1, int dim2) {
    Reshape(std::vector<int>{dim0, dim1, dim2});
  }
  /**
   * Helper for Reshape(const std::vector<int>& shape) with shape.size() = 4.
   *
   * @param[in] dim0, dim0 = shape[0]
   * @param[in] dim1, dim1 = shape[1]
   * @param[in] dim2, dim2 = shape[2]
   * @param[in] dim3, dim3 = shape[3]
   */
  void Reshape(int dim0, int dim1, int dim2, int dim3) {
    Reshape(std::vector<int>{dim0, dim1, dim2, dim3});
  }
  /**
   * Reshape as the shape of *other* Blob.
   * @param[in] other
   */
  void ReshapeLike(const Blob& other);
  /**
   * @brief Copy from a source Blob.
   *
   * @param source the Blob to copy from
   * @param reshape if false, require this Blob to be pre-shaped to the shape
   * of other (and die otherwise); if true, Reshape this Blob to other's
   * shape if necessary
   */
  void CopyFrom(const Blob<Dtype>& source, bool reshape);
  /**
   * call CopyFrom(const Blob<Dtype>& source, bool reshape) with reshape = false
   */
  void CopyFrom(const Blob<Dtype>& source);

  void FromProto(const singa::BlobProto& proto);
  void ToProto(singa::BlobProto* proto) const;
  /**
   * Set each element to be v
   */
  void SetValue(Dtype v);
  /**
   * Compute the sum of absolute values (L1 norm) of the data.
  Dtype AsumData() const;
   */
  /**
   * Sum all elements
  Dtype SumData() const;
   */
  /**
   * Share data with the other Blob.
   * Set the data_ shared_ptr to point to the SyncedMemory holding the data_
   * of Blob other.
   *
   * It may deallocate the SyncedMemory holding this Blob's data_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareData(const Blob& other);

  /*
  void Swap(Blob& other);
  */
  /**
   * @return the shape vector.
   */
  inline const std::vector<int>& shape() const { return shape_; }
  /**
   * @return the size of the k-th dimension.
   */
  inline const int shape(int k) const {
    CHECK_LT(k, shape_.size());
    return shape_.at(k);
  }
  inline int count() const {
    return count_;
  }
  inline const int version() const {
    return version_;
  }
  inline void set_version(int v) {
    version_ = v;
  }
  inline const Dtype* cpu_data() const {
    CHECK(data_);
    return static_cast<const Dtype*>(data_->cpu_data());
  }
  inline void set_cpu_data(Dtype* data) {
    CHECK(data);
    data_->set_cpu_data(data);
  }
  inline const Dtype* gpu_data() const {
    CHECK(data_);
    return static_cast<const Dtype*>(data_->gpu_data());
  }
  inline Dtype* mutable_cpu_data() {
    CHECK(data_);
    return static_cast<Dtype*>(data_->mutable_cpu_data());
  }
  inline Dtype* mutable_gpu_data() {
    CHECK(data_);
    return static_cast<Dtype*>(data_->mutable_gpu_data());
  }
  inline void set_transpose(bool val) {
    transpose_ = val;
  }
  inline bool transpose() const {
    return transpose_;
  }

 protected:
  std::shared_ptr<SyncedMemory> data_ = nullptr;
  std::vector<int> shape_;
  int count_ = 0;
  int capacity_ = 0;
  int version_ = -1;
  bool transpose_ = false;
};  // class Blob

/**
 * Reshape a Blob.
 * @return a new Blob with the given shape, it shares the internal data_ with
 * the original Blob, i.e., no memory copy and allocation.
 */
template <typename Dtype>
Blob<Dtype>* Reshape(const Blob<Dtype> & A, const std::vector<int>& shape) {
  Blob<Dtype>* res = new Blob<Dtype>(A);
  res->Reshape(shape);
  return res;
}

/**
 * Helper of Reshape(const Blob<Dtype>, const std::vector<int>*).
 */
template <typename Dtype>
Blob<Dtype>* Reshape(const Blob<Dtype> & A, int count) {
  std::vector<int> tmpshape;
  tmpshape.push_back(count);
  return Reshape(A, tmpshape);
}
/**
 * Helper of Reshape(const Blob<Dtype>, const std::vector<int>*).
 */
template <typename Dtype>
Blob<Dtype>* Reshape(const Blob<Dtype> & A, int dim0, int dim1) {
  std::vector<int> tmpshape;
  tmpshape.push_back(dim0);
  tmpshape.push_back(dim1);;
  return Reshape(A, tmpshape);
}
/**
 * Helper of Reshape(const Blob<Dtype>, const std::vector<int>*).
 */
template <typename Dtype>
Blob<Dtype>* Reshape(const Blob<Dtype> & A, int dim0, int dim1, int dim2) {
  std::vector<int> tmpshape;
  tmpshape.push_back(dim0);
  tmpshape.push_back(dim1);
  tmpshape.push_back(dim2);
  return Reshape(A, tmpshape);
}
/**
 * Helper of Reshape(const Blob<Dtype>, const std::vector<int>*).
 */
template <typename Dtype>
Blob<Dtype>* Reshape(const Blob<Dtype> & A, int dim0, int dim1, int dim2,
    int dim3) {
  std::vector<int> tmpshape;
  tmpshape.push_back(dim0);
  tmpshape.push_back(dim1);
  tmpshape.push_back(dim2);
  tmpshape.push_back(dim3);
  return Reshape(A, tmpshape);
}

/**
 * @return a new Blob which share all internal members with the input Blob
 * except that the transpose_ field is set to the opposite value.
 */
template <typename Dtype>
Blob<Dtype>* Transpose(const Blob<Dtype> & A) {
  Blob<Dtype>* res = new Blob<Dtype>(A);
  bool origin = A.transpose();
  res->set_transpose(!origin);
  return res;
}

// TODO(wangwei) remove mshadow functions.
using namespace mshadow;
using mshadow::cpu;

using mshadow::Shape;
using mshadow::Shape1;
using mshadow::Shape2;
using mshadow::Shape3;
using mshadow::Shape4;
using mshadow::Tensor;

using std::vector;

inline Tensor<cpu, 4> Tensor4(Blob<float>* blob) {
  const vector<int>& shape = blob->shape();
  Tensor<cpu, 4> tensor(blob->mutable_cpu_data(),
      Shape4(shape[0], shape[1], shape[2], shape[3]));
  return tensor;
}

inline Tensor<cpu, 3> Tensor3(Blob<float>* blob) {
  const vector<int>& shape = blob->shape();
  Tensor<cpu, 3> tensor(blob->mutable_cpu_data(),
      Shape3(shape[0], shape[1], blob->count() / shape[0] / shape[1]));
  return tensor;
}

inline Tensor<cpu, 2> Tensor2(Blob<float>* blob) {
  const vector<int>& shape = blob->shape();
  Tensor<cpu, 2> tensor(blob->mutable_cpu_data(),
      Shape2(shape[0], blob->count() / shape[0]));
  return tensor;
}

inline Tensor<cpu, 1> Tensor1(Blob<float>* blob) {
  Tensor<cpu, 1> tensor(blob->mutable_cpu_data(), Shape1(blob->count()));
  return tensor;
}


}  // namespace singa

#endif  // SINGA_UTILS_BLOB_H_
