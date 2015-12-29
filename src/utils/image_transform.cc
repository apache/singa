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
#include "singa/utils/image_transform.h"

namespace singa {

void ImageTransform(const float* in, const float* mean, bool mirror, int h_crop,
    int w_crop, int h_offset, int w_offset, int channel, int height, int width,
    float scale, float* out) {
  if (h_crop == 0) {
    CHECK_EQ(h_offset, 0);
    h_crop = height;
  }
  if (w_crop ==0) {
    CHECK_EQ(w_offset, 0);
    w_crop = width;
  }
  CHECK_NE(scale, 0);

  int out_idx = 0, in_idx = 0;
  for (int c = 0; c < channel; c++) {
    for (int h = 0; h < h_crop; h++) {
      for (int w = 0; w < w_crop; w++) {
        in_idx = (c * height + h_offset + h) * width + w_offset + w;
        if (mirror) {
          out_idx = (c * h_crop + h) * w_crop + (w_crop - 1 - w);
        } else {
          out_idx = (c * h_crop + h) * w_crop + w;
        }
        out[out_idx] = in[in_idx];
        if (mean != nullptr)
          out[out_idx] -= mean[in_idx];
        out[out_idx] *= scale;
      }
    }
  }
}

}  // namespace singa
