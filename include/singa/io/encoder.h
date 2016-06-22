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
#include "singa/proto/model.pb.h"

namespace singa {
namespace io {

class Encoder {
  public:
    Encoder() { }
    virtual ~Encoder() { }
    
    virtual void Setup(const EncoderConf& conf) = 0;

    /**
     * Format each sample data as a string,
     * whose structure depends on the proto definition.
     * e.g., {key, shape, label, type, data, ...}
     */
    virtual std::string Encode(vector<Tensor>& data) = 0;
};

class JPG2ProtoEncoder : public Encoder {
  public:
    void Setup(const EncoderConf& conf) override;
    std::string Encode(vector<Tensor>& data) override;
};

} // namespace io
} // namespace singa
#endif  // SINGA_IO_ENCODER_H_
