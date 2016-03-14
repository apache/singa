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
 * The code is adapted from Caffe under BSD 2 Clause license.
 *
 * COPYRIGHT
 * All contributions by the University of California:
 * Copyright (c) 2014, The Regents of the University of California (Regents)
 * All rights reserved.
 * All other contributions:
 * Copyright (c) 2014, the respective contributors
 * All rights reserved.
 */
#include "singa/utils/blob.h"

#include <cblas.h>
#include <math.h>
#include <utility>

#define NOT_IMPLEMENTED LOG(FATAL) << "Not implemented function"
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
#include "singa/utils/cuda_utils.h"
#endif  // CPU_ONLY

namespace singa {

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

const void* SyncedMemory::cpu_data() {
  to_cpu();
  return cpu_ptr_;
}

const void* SyncedMemory::gpu_data() {
#ifndef CPU_ONLY
  to_gpu();
  return gpu_ptr_;
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

void SyncedMemory::set_cpu_data(void* data) {
  CHECK(data);
  if (own_cpu_data_) {
    FreeHost(cpu_ptr_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

void SyncedMemory::to_cpu() {
  switch (head_) {
  case UNINITIALIZED:
    MallocHost(&cpu_ptr_, size_);
    memset(cpu_ptr_, 0, size_);
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

void SyncedMemory::to_gpu() {
#ifndef CPU_ONLY
  switch (head_) {
  case UNINITIALIZED:
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    CUDA_CHECK(cudaMemset(gpu_ptr_, 0, size_));
    head_ = HEAD_AT_GPU;
    break;
  case HEAD_AT_CPU:
    if (gpu_ptr_ == NULL) {
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    }
    CUDA_CHECK(cudaMemcpy(gpu_ptr_, cpu_ptr_, size_, cudaMemcpyDefault));
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

template <typename Dtype>
void Blob<Dtype>::Reshape(const std::vector<int>& shape) {
  shape_ = shape;
  count_ = shape.size() ? 1 : 0;
  for (size_t i = 0; i < shape.size(); ++i) {
    CHECK(shape[i]);
    count_ *= shape[i];
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
void Blob<Dtype>::CopyFrom(const Blob& source) {
    CopyFrom(source, false);
}

template <typename Dtype>
void Blob<Dtype>::CopyFrom(const Blob& source, bool shape_check) {
  LOG(WARNING) << "Better use Copy(const Blob&, Blob*)";
  CHECK_EQ(source.count(), count()) << " cp between blobs of diff size";

  if (shape_check &&
      !std::equal(shape_.begin(), shape_.end(), source.shape_.begin())) {
      LOG(FATAL) << "Trying to copy blobs of different sizes.";
  }
#ifndef CPU_ONLY
  CUDA_CHECK(cudaMemcpy(static_cast<Dtype*>(data_->mutable_gpu_data()),
             source.gpu_data(), sizeof(Dtype) * count_, cudaMemcpyDefault));
#endif
  memcpy(static_cast<Dtype*>(data_->mutable_cpu_data()), source.cpu_data(),
         sizeof(Dtype)*count_);
}

template <typename Dtype>
void Blob<Dtype>::FromProto(const singa::BlobProto& proto) {
  std::vector<int> shape;
  for (int s : proto.shape()) {
    shape.push_back(s);
  }
  int count = count_;
  Reshape(shape);
  if (count != count_)
    LOG(WARNING) << "Blob is reshaped to diff size " << count << ":" << count_;
  // copy data
  Dtype* data_vec = mutable_cpu_data();
  for (int i = 0; i < count_; ++i) {
    data_vec[i] = proto.data(i);
  }
}

template <typename Dtype>
void Blob<Dtype>::ToProto(singa::BlobProto* proto) const {
  for (int s : shape_) {
    proto->add_shape(s);
  }
  proto->clear_data();
  const Dtype* data_vec = cpu_data();
  for (int i = 0; i < count_; ++i) {
    proto->add_data(data_vec[i]);
  }
}

template <typename Dtype>
void Blob<Dtype>::SetValue(Dtype v) {
  Dtype* ptr = mutable_cpu_data();
  for (int i =0; i < count(); i++)
    ptr[i] = v;
}
template <typename Dtype>
void Blob<Dtype>::ShareData(Blob* other, bool cpu_only) {
  CHECK_EQ(count_, other->count());
  if (cpu_only)
    data_->set_cpu_data(other->mutable_cpu_data());
  else
    data_ = other->data_;
}

/*
template <typename Dtype>
void Blob<Dtype>::Swap(Blob& other) {
  CHECK_EQ(other.count(), count());
  CHECK(std::equal(shape_.begin(), shape_.end(), other.shape_.begin()));
  std::swap(data_, other.data_);
  std::swap(capacity_, other.capacity_);
}
*/

INSTANTIATE_CLASS(Blob);
template class Blob<int>;
template class Blob<unsigned int>;

}  // namespace singa
