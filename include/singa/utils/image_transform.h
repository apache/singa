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

#ifndef SINGA_UTILS_IMAGE_TRANSFORM_H_
#define SINGA_UTILS_IMAGE_TRANSFORM_H_

#include <glog/logging.h>
// TODO(wangwei) provide image transformation API, the implementation can be
// done by opencv, manual transform, or mshadow.
namespace singa {

void ImageTransform(const float* in, const float* mean, bool mirror, int h_crop,
    int w_crop, int h_offset, int w_offset, int channel, int height, int width,
    float scale, float* out);
}  // namespace singa

#endif  // SINGA_UTILS_IMAGE_TRANSFORM_H_
