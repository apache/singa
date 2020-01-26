/*********************************************************
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
************************************************************/
#ifndef SINGA_UTILS_MKLDNN_UTILS_H_
#define SINGA_UTILS_MKLDNN_UTILS_H_

#include <mkldnn.hpp>

namespace singa {
/*
 supported data type by mkldnn
 mkldnn_f32 - 32-bit/single-precision floating point.
 mkldnn_s32 - 32-bit signed integer.
 mkldnn_s16 - 16-bit signed integer.
 mkldnn_s8 - 8-bit signed integer.
 mkldnn_u8 - 8-bit unsigned integer.
 */
inline mkldnn::memory::data_type GetMKLDNNDataType(DataType dtype) {
  mkldnn::memory::data_type ret = mkldnn::memory::data_type::f32;
  switch (dtype) {
    case kFloat32:
      ret = mkldnn::memory::data_type::f32;
      break;
    case kDouble:
      LOG(FATAL) << "The data type " << DataType_Name(dtype)
                 << " is not support by mkldnn";
      break;
    case kFloat16:
      LOG(FATAL) << "The data type " << DataType_Name(dtype)
                 << " is not support by mkldnn";
      break;
    default:
      LOG(FATAL) << "The data type " << DataType_Name(dtype)
                 << " is not support by mkldnn";
  }
  return ret;
}
}
#endif  // SINGA_UTILS_MKLDNN_UTILS_H_
