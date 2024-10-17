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

#ifndef SINGA_MODEL_INITIALIZER_H_
#define SINGA_MODEL_INITIALIZER_H_
#include <string>

#include "singa/core/tensor.h"
#include "singa/proto/model.pb.h"
#include "singa/utils/string.h"
namespace singa {
/// Base class for initializing parameter values.
using InitializerConf = FillerConf;
class Initializer {
 public:
  Initializer() = default;
  void Setup(const std::string& str) {
    InitializerConf conf;
    conf.ParseFromString(str);
    Setup(conf);
  }

  /// Set meta fields from user configurations.
  virtual void Setup(const InitializerConf& conf) {}

  virtual void Fill(Tensor& t) = 0;
};

namespace init {
class Constant : public Initializer {
 public:
  Constant() = default;
  Constant(const float x) : v_(x) {}
  void Setup(const InitializerConf& conf) override { v_ = conf.value(); }
  void Fill(Tensor& t) override { t.SetValue(v_); }

 private:
  float v_ = 0;
};

class Uniform : public Initializer {
 public:
  Uniform() = default;
  Uniform(const float low, const float high) : min_(low), max_(high) {}
  void Setup(const InitializerConf& conf) override {
    min_ = conf.min();
    max_ = conf.max();
  }
  void Fill(Tensor& t) override { singa::Uniform(min_, max_, &t); }

 private:
  float min_ = 0, max_ = 1;
};

class Gaussian : public Initializer {
 public:
  Gaussian() = default;
  Gaussian(const float m, const float s) : mean_(m), std_(s) {}
  void Setup(const InitializerConf& conf) override {
    mean_ = conf.mean();
    std_ = conf.std();
  }
  void Fill(Tensor& t) override { singa::Gaussian(mean_, std_, &t); }

 private:
  float mean_ = 0, std_ = 1;
};

/// Ref: [Bengio and Glorot 2010] Understanding the difficulty of training deep
/// feedforward neural networks
class Xavier : public Initializer {
 public:
  void Fill(Tensor& t) override {
    CHECK_EQ(t.nDim(), 2u);
    float scale = sqrt(6.0f / (t.shape(0) + t.shape(1)));
    LOG(INFO) << "xavier scale " << scale;
    singa::Uniform(-scale, scale, &t);
  }
};

/// Ref: [He, Zhang, Ren and Sun 2015]: Delving Deep into Rectifiers:
/// Surpassing Human-Level Performance on ImageNet Classification
class MSRA : public Initializer {
 public:
  void Fill(Tensor& t) override {
    CHECK_EQ(t.nDim(), 2u);
    float std = sqrt(2.0f / t.shape(0));
    singa::Gaussian(0.0f, std, &t);
  }
};

}  // namespace init

/// TODO(wangwei) create the initializers from factory like that for Layer.
std::shared_ptr<Initializer> CreateInitializer(const InitializerConf& conf) {
  std::shared_ptr<Initializer> init;
  if (ToLowerCase(conf.type()) == "constant") {
    init = std::make_shared<init::Constant>();
  } else if (ToLowerCase(conf.type()) == "uniform") {
    init = std::make_shared<init::Uniform>();
  } else if (ToLowerCase(conf.type()) == "gaussian") {
    init = std::make_shared<init::Gaussian>();
  } else if (ToLowerCase(conf.type()) == "xavier") {
    init = std::make_shared<init::Xavier>();
  } else if (ToLowerCase(conf.type()) == "msra") {
    init = std::make_shared<init::MSRA>();
  } else {
    LOG(FATAL) << "Unknown initialization type: " << conf.type();
  }
  init->Setup(conf);
  return init;
}
}  // namespace singa
#endif  // SINGA_MODEL_INITIALIZER_H_
