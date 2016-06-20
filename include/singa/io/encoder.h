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

#ifndef SINGA_IO_ENCODER_H_
#define SINGA_IO_ENCODER_H_

#include <vector>
#include <string>
#include "singa/core/tensor.h"

namespace singa {
namespace io {

class Encoder {
  public:
    Encoder() { }
    virtual ~Encoder() { }

    /**
     * Format each sample data as a string,
     * whose structure depends on the proto definition.
     * e.g., {key, shape, label, type, data, ...}
     */
    virtual std::string Encode(vector<Tensor>& data) { return ""; }
};

} // namespace io
} // namespace singa
#endif  // SINGA_IO_ENCODER_H_
