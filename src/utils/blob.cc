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
 */
#include <utility>
#include <math.h>
#include <cblas.h>
#include "utils/blob.h"
/*********************SyncedMemory implementation************************/

#define NO_GPU LOG(FATAL) << "CPU-only Mode: cannot make GPU call."
// Instantiate a class with float and double specifications.
#define INSTANTIATE_CLASS(classname) \
  template class classname<float>; \
  template class classname<double>
// Disable the copy and assignment operator for a class.
#define DISABLE_COPY_AND_ASSIGN(classname) \
private:\
  classname(const classname&);\
  classname& operator=(const classname&)

#ifndef CPU_ONLY
// CUDA: various checks for different function calls.
#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error); \
  } while (0)

#define CUBLAS_CHECK(condition) \
  do { \
    cublasStatus_t status = condition; \
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " " \
      << caffe::cublasGetErrorString(status); \
  } while (0)

#define CURAND_CHECK(condition) \
  do { \
    curandStatus_t status = condition; \
    CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " " \
      << caffe::curandGetErrorString(status); \
  } while (0)

#endif // CPU_ONLY


SyncedMemory::~SyncedMemory() {
  if (cpu_ptr_ && own_cpu_data_) {
    FreeHost(cpu_ptr_);
  }

#ifndef CPU_ONLY
  if (gpu_ptr_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
#endif  // CPU_ONLY
}

inline void SyncedMemory::to_cpu() {
  switch (head_) {
  case UNINITIALIZED:
    MallocHost(&cpu_ptr_, size_);
    memset(cpu_ptr_,0, size_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:
#ifndef CPU_ONLY
    if (cpu_ptr_ == NULL) {
      MallocHost(&cpu_ptr_, size_);
      own_cpu_data_ = true;
    }
    CUDA_CHECK(cudaMemcpy(cpu_ptr_, gpu_ptr_, size_, cudaMemcpyDefault));
    head_ = SYNCED;
#else
    NO_GPU;
#endif
    break;
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

inline void SyncedMemory::to_gpu() {
#ifndef CPU_ONLY
  switch (head_) {
  case UNINITIALIZED:
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    CUDA_CHECK(cudaMemset(gpu_ptr_, 0, N));  // NOLINT(caffe/alt_fn)
    head_ = HEAD_AT_GPU;
    break;
  case HEAD_AT_CPU:
    if (gpu_ptr_ == NULL) {
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    }
    CUDA_CHECK(cudaMemcpy( gpu_ptr_,cpu_ptr_, size_, cudaMemcpyDefault));
    head_ = SYNCED;
    break;
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
#else
  NO_GPU;
#endif
}

const void* SyncedMemory::cpu_data() {
  to_cpu();
  return (const void*)cpu_ptr_;
}

void SyncedMemory::set_cpu_data(void* data) {
  CHECK(data);
  if (own_cpu_data_) {
    FreeHost(cpu_ptr_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

const void* SyncedMemory::gpu_data() {
#ifndef CPU_ONLY
  to_gpu();
  return (const void*)gpu_ptr_;
#else
  NO_GPU;
#endif
  return nullptr;
}

void* SyncedMemory::mutable_cpu_data() {
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data() {
#ifndef CPU_ONLY
  to_gpu();
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
#else
  NO_GPU;
#endif
  return nullptr;
}

/*********************Blob implementation************************/

template <typename Dtype>
Blob<Dtype>::Blob(const vector<int>& shape)
  // capacity_ must be initialized before calling Reshape
  : capacity_(0) {
  Reshape(shape);
}

template <typename Dtype>
void Blob<Dtype>::Reshape(const vector<int>& shape) {
  count_=1;
  shape_=shape;
  for(size_t i=0;i<shape.size();i++){
    CHECK(shape[i]);
    count_*=shape[i];
  }
  if (count_ > capacity_) {
    capacity_ = count_;
    data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
  }
}

template <typename Dtype>
void Blob<Dtype>::ReshapeLike(const Blob<Dtype>& other) {
  Reshape(other.shape());
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->cpu_data();
}

template <typename Dtype>
void Blob<Dtype>::set_cpu_data(Dtype* data) {
  CHECK(data);
  data_->set_cpu_data(data);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_data() const {
  CHECK(data_);
  return (const Dtype*)data_->gpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data() {
  CHECK(data_);
  return static_cast<Dtype*>(data_->mutable_gpu_data());
}

template <typename Dtype>
void Blob<Dtype>::ShareData(const Blob& other) {
  CHECK_EQ(count_, other.count());
  data_ = other.data();
}

template <> float Blob<float>::asum_data() const {
  if(count()==0)
    return 0.f;
  return cblas_sasum(count(), cpu_data(), 1)/count();
}
template <> float Blob<float>::sum_data() const {
  if(count()==0)
    return 0.f;
  float sum=0.f;
  const float *dptr=cpu_data();
  for(int i=0;i<count();i++)
    sum+=dptr[i];
  return sum/count();
}
template <> unsigned int Blob<unsigned int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <> int Blob<int>::asum_data() const {
  NOT_IMPLEMENTED;
  return 0;
}

template <typename Dtype>
void Blob<Dtype>::Swap(Blob& other){
  CHECK_EQ(other.count(), count());
  CHECK(std::equal(shape_.begin(), shape_.end(), other.shape_.begin()));
  std::swap(data_, other.data_);
  std::swap(capacity_, other.capacity_);
}

template <typename Dtype>
void Blob<Dtype>::CopyFrom(const Blob& source, bool reshape) {
  if (!std::equal(shape_.begin(),shape_.end(),source.shape_.begin())) {
    if (reshape) {
      Reshape(source.shape_);
    } else {
      LOG(FATAL) << "Trying to copy blobs of different sizes.";
    }
  }
#ifndef CPU_ONLY
  CUDA_CHECK(cudaMemcpy(static_cast<Dtype*>(data_->mutable_gpu_data()),
            source.gpu_data(), sizeof(Dtype) * count_, cudaMemcpyDefault));
#endif
  memcpy(static_cast<Dtype*>(data_->mutable_cpu_data()),source.cpu_data(),
        sizeof(Dtype)*count_);
}

/*
template <typename Dtype>
void Blob<Dtype>::FromProto(const BlobProto& proto) {
  Reshape();
  // copy data
  Dtype* data_vec = mutable_cpu_data();
  for (int i = 0; i < count_; ++i) {
    data_vec[i] = proto.data(i);
  }
}
*/

template <typename Dtype>
void Blob<Dtype>::ToProto(singa::BlobProto* proto) const {
  proto->set_num(shape_[0]);
  if(shape_.size()>1)
    proto->set_channels(shape_[1]);
  if(shape_.size()>2)
    proto->set_height(shape_[2]);
  if(shape_.size()>3)
    proto->set_width(shape_[3]);
  proto->clear_data();
  const Dtype* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_data(data_vec[i]);
  }
}

INSTANTIATE_CLASS(Blob);
template class Blob<int>;
template class Blob<unsigned int>;
