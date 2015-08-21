#ifndef SINGA_UTILS_UPDATER_H_
#define SINGA_UTILS_UPDATER_H_

#include "proto/job.pb.h"
#include "utils/param.h"

namespace singa {
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
  static Updater* Create(const UpdaterProto& proto);

  virtual ~Updater() {}

  virtual void Init(const UpdaterProto &proto);
  virtual void Update(int step, Param* param, float grad_scale) = 0;

 protected:
  UpdaterProto proto_;
  LRGenerator* lr_gen_;
  float weight_decay_;
  float momentum_;
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

/*
class RMSPropUpdater : public Updater {
 public:
  virtual void Update(int step, Param* param, float grad_scale);

 protected:
  float base_lr_;
  float delta_;
  float rho_;
  float weight_decay_;
};

class AdaDeltaUpdater : public Updater {
 public:
  virtual void Update(int step, Param* param, float grad_scale);

 protected:
  float rho_;
  float delta_;
  float weight_decay_;
};
*/

}  // namespace singa

#endif  // SINGA_UTILS_UPDATER_H_
