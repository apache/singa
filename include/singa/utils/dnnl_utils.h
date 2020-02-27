/*********************************************************
 * *
 * * Licensed to the Apache Software Foundation (ASF) under one
 * * or more contributor license agreements.  See the NOTICE file
 * * distributed with this work for additional information
 * * regarding copyright ownership.  The ASF licenses this file
 * * to you under the Apache License, Version 2.0 (the
 * * "License"); you may not use this file except in compliance
 * * with the License.  You may obtain a copy of the License at
 * *
 * *   http://www.apache.org/licenses/LICENSE-2.0
 * *
 * * Unless required by applicable law or agreed to in writing,
 * * software distributed under the License is distributed on an
 * * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * * KIND, either express or implied.  See the License for the
 * * specific language governing permissions and limitations
 * * under the License.
 * *
 * ************************************************************/
#ifndef SINGA_UTILS_MKLDNN_UTILS_H_
#define SINGA_UTILS_MKLDNN_UTILS_H_

namespace singa {

using namespace dnnl;

inline dnnl::memory::format_tag get_dnnl_format_tag(const Tensor &x) {
  memory::format_tag format_tag_;
  switch (x.nDim()) {
    case 1: {
      format_tag_ = memory::format_tag::a;
      break;
    }
    case 2: {
      format_tag_ = memory::format_tag::ab;
      break;
    }
    case 3: {
      format_tag_ = memory::format_tag::abc;
      break;
    }
    case 4: {
      format_tag_ = memory::format_tag::abcd;
      break;
    }
    default: {
      LOG(FATAL) << x.nDim() << " dim is not supported";
    }
  }
  return format_tag_;
}
}  // namespace singa
#endif  // SINGA_UTILS_MKLDNN_UTILS_H_
