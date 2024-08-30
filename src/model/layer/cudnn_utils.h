/*
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
#ifndef SRC_MODEL_LAYER_CUDNN_UTILS_H_
#define SRC_MODEL_LAYER_CUDNN_UTILS_H_

#include "singa/singa_config.h"
#ifdef USE_CUDNN

#include <cudnn.h>

#include "singa/proto/core.pb.h"
#include "singa/utils/logging.h"
namespace singa {
inline cudnnDataType_t GetCudnnDataType(DataType dtype) {
  cudnnDataType_t ret = CUDNN_DATA_FLOAT;
  switch (dtype) {
    case kFloat32:
      ret = CUDNN_DATA_FLOAT;
      break;
    case kDouble:
      ret = CUDNN_DATA_DOUBLE;
      break;
    case kFloat16:
      ret = CUDNN_DATA_HALF;
      break;
    default:
      LOG(FATAL) << "The data type " << DataType_Name(dtype)
                 << " is not support by cudnn";
  }
  return ret;
}

#define CUDNN_CHECK(condition)                 \
  do {                                         \
    cudnnStatus_t status = condition;          \
    CHECK_EQ(status, CUDNN_STATUS_SUCCESS)     \
        << " " << cudnnGetErrorString(status); \
  } while (0)

/*
inline const char* cudnnGetErrorString(cudnnStatus_t status) {
  switch (status) {
    case CUDNN_STATUS_SUCCESS:
      return "CUDNN_STATUS_SUCCESS";
    case CUDNN_STATUS_NOT_INITIALIZED:
      return "CUDNN_STATUS_NOT_INITIALIZED";
    case CUDNN_STATUS_ALLOC_FAILED:
      return "CUDNN_STATUS_ALLOC_FAILED";
    case CUDNN_STATUS_BAD_PARAM:
      return "CUDNN_STATUS_BAD_PARAM";
    case CUDNN_STATUS_INTERNAL_ERROR:
      return "CUDNN_STATUS_INTERNAL_ERROR";
    case CUDNN_STATUS_INVALID_VALUE:
      return "CUDNN_STATUS_INVALID_VALUE";
    case CUDNN_STATUS_ARCH_MISMATCH:
      return "CUDNN_STATUS_ARCH_MISMATCH";
    case CUDNN_STATUS_MAPPING_ERROR:
      return "CUDNN_STATUS_MAPPING_ERROR";
    case CUDNN_STATUS_EXECUTION_FAILED:
      return "CUDNN_STATUS_EXECUTION_FAILED";
    case CUDNN_STATUS_NOT_SUPPORTED:
      return "CUDNN_STATUS_NOT_SUPPORTED";
    case CUDNN_STATUS_LICENSE_ERROR:
      return "CUDNN_STATUS_LICENSE_ERROR";
  }
  return "Unknown cudnn status";
}
*/

}  // namespace singa
#endif  // USE_CUDNN
#endif  // SRC_MODEL_LAYER_CUDNN_UTILS_H_
