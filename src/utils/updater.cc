
#include "utils/updater.h"
#include "mshadow/tensor.h"
#include "mshadow/cxxnet_op.h"
#include "proto/job.pb.h"
using namespace mshadow;
using namespace mshadow::expr;

namespace  singa {

float Updater::GetLearningRate(int step){
  float ret = 0., r = 0., base=proto_.base_lr();
  int freq=0;
  switch (proto_.lr_change()) {
    case UpdaterProto_ChangeMethod_kFixed:
      ret = base;
      break;
    case UpdaterProto_ChangeMethod_kLinear:
      // a is init, b is the final
      freq=proto_.linear_conf().change_freq();
      r = step * 1.0  / freq;
      ret = (1.0 - r) * base + r * proto_.linear_conf().final_lr();
      break;
    case UpdaterProto_ChangeMethod_kExponential:
      // a is init, b is the final, from convnet
      freq=proto_.exponential_conf().change_freq();
      ret = base / pow(2, step * 1. / freq);
      break;
    case UpdaterProto_ChangeMethod_kInverseT:
      // a is init, b is the final, from convnet
      CHECK_EQ(base, 2 * proto_.inverset_conf().final_lr())
        << "final value should be the half";
      ret = base / (1. + step * 1. / proto_.inverset_conf().final_lr());
      break;
    case UpdaterProto_ChangeMethod_kInverse:
      // a is init, b is gamma, c is pow
      ret=base*pow(1.f+proto_.inverse_conf().gamma()*step,
          -proto_.inverse_conf().pow());
      break;
    case UpdaterProto_ChangeMethod_kStep:
      // a is the base learning rate, b is gamma, from caffe
      // notice it is step/change_steps, not step*1.0/change_steps
      freq=proto_.step_conf().change_freq();
      ret = base * pow(proto_.step_conf().gamma(), step / freq);
      break;
    case UpdaterProto_ChangeMethod_kFixedStep:
      for(int i=0;i<proto_.fixedstep_conf().step_size();i++){
        if(step>proto_.fixedstep_conf().step(i))
          ret=proto_.fixedstep_conf().step_lr(i);
      }
      break;
    default:
      LOG(ERROR) << "Wrong hyper-parameter update method";
  }
  return ret;
}

/***********************SGD with momentum******************************/
void SGDUpdater::Init(const UpdaterProto& proto){
  Updater::Init(proto);
  base_lr_=proto.base_lr();
  //CHECK_GT(base_lr_, 0);
  momentum_=proto.momentum();
  weight_decay_=proto.weight_decay();
}

void SGDUpdater::Update(int step, Param* param, float grad_scale){
  Shape<1> s=Shape1(param->size());
  Tensor<cpu, 1> data(param->mutable_cpu_data(), s);
  Tensor<cpu, 1> grad(param->mutable_cpu_grad(), s);
  float lr=GetLearningRate(step)*param->learning_rate_multiplier();
  float wd=weight_decay_*param->weight_decay_multiplier();
  if(wd>0){ // L2 regularization
    grad+=data*wd;
  }
  if(momentum_>0){
    Tensor<cpu, 1> history(param->mutable_cpu_history(), s);
    if(step==0) history=0;
    history=history*momentum_-lr*grad;
    data+=history;
  }else{
    grad*=-lr;
    data+=grad;
  }
}

/***********************Nesterov******************************/
void NesterovUpdater::Init(const UpdaterProto& proto){
  Updater::Init(proto);
  base_lr_=proto.base_lr();
  CHECK_GT(base_lr_, 0);
  weight_decay_=proto.weight_decay();
}

void NesterovUpdater::Update(int step, Param* param, float grad_scale){
  Shape<1> s=Shape1(param->size());
  Tensor<cpu, 1> data(param->mutable_cpu_data(), s);
  Tensor<cpu, 1> grad(param->mutable_cpu_grad(), s);
  Tensor<cpu, 1> history(param->mutable_cpu_history(), s);
  TensorContainer<cpu, 1> tmp(s);
  if(step==0) history=0;
  float lr=GetLearningRate(step)*param->learning_rate_multiplier();
  float wd=weight_decay_*param->weight_decay_multiplier();
  if(wd>0){ // L2 regularization
    grad+=data*wd;
  }
  Copy(tmp, history);
  history=history*momentum_+lr*grad;
  tmp=history*(1+momentum_)-tmp*momentum_;
  data-=tmp;
}
/***********************AdaGrad******************************/
void AdaGradUpdater::Init(const UpdaterProto& proto){
  Updater::Init(proto);
  base_lr_=proto.base_lr();
  CHECK_GT(base_lr_, 0);
  delta_=proto.delta();
  weight_decay_=proto.weight_decay();
}

void AdaGradUpdater::Update(int step, Param* param, float grad_scale){
  Shape<1> s=Shape1(param->size());
  Tensor<cpu, 1> data(param->mutable_cpu_data(), s);
  Tensor<cpu, 1> grad(param->mutable_cpu_grad(), s);
  Tensor<cpu, 1> history(param->mutable_cpu_history(), s);
  if(step==0) history=0;
  history+=F<op::square>(grad*grad_scale);
  float lr=GetLearningRate(step)*param->learning_rate_multiplier();
  float wd=weight_decay_*param->weight_decay_multiplier();
  if(wd>0){ // L2 regularization
    grad+=data*wd;
  }
  data-=lr*grad/(F<op::sqrtop>(history,delta_));
}

/***********************RMSProp******************************/
void RMSPropUpdater::Init(const UpdaterProto& proto){
  Updater::Init(proto);
  base_lr_=proto.base_lr();
  CHECK_GT(base_lr_, 0);
  delta_=proto.delta();
  rho_=proto.rmsprop_conf().rho();
  weight_decay_=proto.weight_decay();
}

void RMSPropUpdater::Update(int step, Param* param, float grad_scale){
  Shape<1> s=Shape1(param->size());
  Tensor<cpu, 1> data(param->mutable_cpu_data(), s);
  Tensor<cpu, 1> grad(param->mutable_cpu_grad(), s);
  Tensor<cpu, 1> history(param->mutable_cpu_history(), s);
  if(step==0) history=0;
  history=history*rho_+(1-rho_)*F<op::square>(grad*grad_scale);
  float lr=GetLearningRate(step)*param->learning_rate_multiplier();
  float wd=weight_decay_*param->weight_decay_multiplier();
  if(wd>0){ // L2 regularization
    grad+=data*wd;
  }
  data-=lr*grad/(F<op::sqrtop>(history,delta_));
}

/***********************AdaDelta******************************
void AdaDeltaUpdater::Init(const UpdaterProto& proto){
  Updater::Init(proto);
  delta_=proto.delta();
  rho_=proto.rho();
  weight_decay_=proto.weight_decay();
}

void AdaDeltaUpdater::Update(int step, Param* param, float grad_scale){
  Shape<1> s=Shape1(param->size());
  Tensor<cpu, 1> data(param->mutable_cpu_data(), s);
  Tensor<cpu, 1> grad(param->mutable_cpu_grad(), s);
  Tensor<cpu, 1> history(param->mutable_cpu_history(), s);
  Tensor<cpu, 1> update(param->mutable_cpu_update(), s);
  TensorContainer<cpu, 1> tmp(s);
  float wd=weight_decay_*param->weight_decay_multiplier();
  if(wd>0){ // L2 regularization
    grad+=data*wd;
  }
  if(step==0){
    history=0;
    update=0;
  }
  history=history*rho_+(1-rho_)*F<op::square>(grad*grad_scale);
  tmp=grad*F<op::sqrtop>(update, delta_)/F<op::sqrtop>(history, delta_);
  update=rho_*update+(1-rho_)*F<op::square>(tmp);
  data-=tmp;
}
*/

} /* singa */
