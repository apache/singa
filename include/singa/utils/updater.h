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

#ifndef SINGA_UTILS_UPDATER_H_
#define SINGA_UTILS_UPDATER_H_

#include <string>
#include "singa/proto/job.pb.h"
#include "singa/utils/param.h"
#include "singa/neuralnet/layer.h"

namespace singa {
using std::string;
/**
 * Base learning rate generator.
 *
 * Generate learning rate for a give training step/iteration.
 * There are many different ways to change the learning rate through time/step.
 * Users can inherint this class to implement their own change method.
 */
class LRGenerator {
 public:
  static LRGenerator* Create(const LRGenProto& proto);

  virtual ~LRGenerator() {}

  virtual void Init(const LRGenProto& proto) { proto_ = proto; }
  /**
   * @param step training step/iteration.
   * @return base learning rate regardless of step
   */
  virtual float Get(int step) { return proto_.base_lr(); }

 protected:
  LRGenProto proto_;
};

class FixedStepLRGen : public LRGenerator {
 public:
  float Get(int step) override;
 private:
  int last_idx_ = 0;
};

class StepLRGen : public LRGenerator {
 public:
  float Get(int step) override;
};

class LinearLRGen : public LRGenerator {
 public:
  float Get(int step) override;
};

class ExpLRGen : public LRGenerator {
 public:
  float Get(int step) override;
};

class InvLRGen : public LRGenerator {
 public:
  float Get(int step) override;
};

class InvTLRGen : public LRGenerator {
 public:
  float Get(int step) override;
};

/**
 * Updater for Param.
 */
class Updater {
 public:

  /* added for python binding */
  static Updater* CreateUpdater(const string str);
  /* ------------------------ */

  static Updater* Create(const UpdaterProto& proto);

  virtual ~Updater() {}

  virtual void Init(const UpdaterProto &proto);
  virtual void Update(int step, Param* param, float grad_scale) = 0;
  void Clip(const float low, const float high, Param* param);
 protected:
  UpdaterProto proto_;
  LRGenerator* lr_gen_;
  float weight_decay_;
  float momentum_;
  float clip_low_, clip_high_;
};

class SGDUpdater : public Updater {
 public:
  void Update(int step, Param* param, float grad_scale) override;
};

class AdaGradUpdater : public Updater {
 public:
  void Update(int step, Param* param, float grad_scale) override;
};


class NesterovUpdater : public Updater {
 public:
  void Update(int step, Param* param, float grad_scale) override;
};

class RMSPropUpdater : public Updater {
 public:
  void Init(const UpdaterProto &proto) override;
  void Update(int step, Param* param, float grad_scale) override;

 protected:
  float rho_;
  float delta_;
};

class AdaDeltaUpdater : public Updater {
 public:
  void Init(const UpdaterProto &proto) override;
  void Update(int step, Param* param, float grad_scale) override;

 protected:
  float rho_;
  float delta_;
};

class AdamUpdater : public Updater {
  public:
   void Init(const UpdaterProto &proto) override;
   void Update(int step, Param* param, float grad_scale) override;

  protected:
   float beta1_;
   float beta2_;
   float delta_;
};

class AdamMaxUpdater : public Updater {
  public:
   void Init(const UpdaterProto &proto) override;
   void Update(int step, Param* param, float grad_scale) override;

  protected:
   float beta1_;
   float beta2_;
   float delta_;
};

}  // namespace singa

#endif  // SINGA_UTILS_UPDATER_H_
