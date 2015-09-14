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
#include "proto/common.pb.h"

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
  explicit Blob(const std::vector<int>& shape) { Reshape(shape); }
  /**
   * @brief Change the dimensions of the blob, allocating new memory if
   *        necessary.
   *
   * This function can be called both to create an initial allocation
   * of memory, and to adjust the dimensions of a top blob during Layer::Reshape
   * or Layer::Forward. When changing the size of blob, memory will only be
   * reallocated if sufficient memory does not already exist, and excess memory
   * will never be freed.
   *
   * Note that reshaping an input blob and immediately calling Net::Backward is
   * an error; either Net::Forward or Net::Reshape need to be called to
   * propagate the new input shape to higher layers.
   */
  void Reshape(const std::vector<int>& shape);
  void ReshapeLike(const Blob& other);
  /**
   * @brief Copy from a source Blob.
   *
   * @param source the Blob to copy from
   * @param reshape if false, require this Blob to be pre-shaped to the shape
   *        of other (and die otherwise); if true, Reshape this Blob to other's
   *        shape if necessary
   */
  void CopyFrom(const Blob<Dtype>& source);
  void CopyFrom(const Blob<Dtype>& source, bool reshape);
  void FromProto(const singa::BlobProto& proto);
  void ToProto(singa::BlobProto* proto) const;
  /**
   * @brief Set the data_ shared_ptr to point to the SyncedMemory holding the
   *        data_ of Blob other -- useful in Layer&s which simply perform a copy
   *        in their Forward pass.
   *
   * This deallocates the SyncedMemory holding this Blob's data_, as
   * shared_ptr calls its destructor when reset with the "=" operator.
   */
  void ShareData(const Blob& other);
  void Swap(Blob& other);
  inline const std::vector<int>& shape() const { return shape_; }
  inline int count() const { return count_; }
  inline const int version() const { return version_; }
  inline void set_version(int v) { version_ = v; }
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
  inline Dtype* mutable_xpu_data() {
    CHECK(data_);
	#ifndef CPU_ONLY
		return static_cast<Dtype*>(data_->mutable_gpu_data());
	#else
	    return static_cast<Dtype*>(data_->mutable_cpu_data());
	#endif
  }
  /// @brief Compute the sum of absolute values (L1 norm) of the data.
  Dtype asum_data() const;
  Dtype sum_data() const;

 protected:
  std::shared_ptr<SyncedMemory> data_ = nullptr;
  std::vector<int> shape_;
  int count_ = 0;
  int capacity_ = 0;
  int version_ = -1;
};  // class Blob

}  // namespace singa

#endif  // SINGA_UTILS_BLOB_H_
