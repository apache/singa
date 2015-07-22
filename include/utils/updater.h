#ifndef INCLUDE_UTILS_UPDATER_H_
#define INCLUDE_UTILS_UPDATER_H_
#include "proto/job.pb.h"
#include "utils/param.h"

namespace singa{
/**
 * Updater for Param.
 */
class Updater{
 public:
  virtual ~Updater() {}
  virtual void Init(const UpdaterProto &proto){
    proto_=proto;
  }
  virtual void Update(int step, Param* param, float grad_scale=1.0f)=0;

  float GetLearningRate(int step);
 protected:
  UpdaterProto proto_;
};
class SGDUpdater : public Updater{
 public:
  virtual void Init(const UpdaterProto& proto);
  virtual void Update(int step, Param* param, float grad_scale=1.0f);

 protected:
  float base_lr_;
  float momentum_;
  float weight_decay_;
};
class NesterovUpdater : public Updater{
 public:
  virtual void Init(const UpdaterProto& proto);
  virtual void Update(int step, Param* param, float grad_scale=1.0f);

 protected:
  float base_lr_;
  float momentum_;
  float weight_decay_;
};
class AdaGradUpdater : public Updater{
 public:
  virtual void Init(const UpdaterProto& proto);
  virtual void Update(int step, Param* param, float grad_scale=1.0f);

 protected:
  float base_lr_;
  float delta_;
  float weight_decay_;
};

class RMSPropUpdater : public Updater{
 public:
  virtual void Init(const UpdaterProto& proto);
  virtual void Update(int step, Param* param, float grad_scale=1.0f);

 protected:
  float base_lr_;
  float delta_;
  float rho_;
  float weight_decay_;
};

/*
class AdaDeltaUpdater : public Updater{
 public:
  virtual void Init(const UpdaterProto& proto);
  virtual void Update(int step, Param* param, float grad_scale=1.0f);

 protected:
  float rho_;
  float delta_;
  float weight_decay_;
};
*/
}

#endif // INCLUDE_UTILS_UPDATER_H_
