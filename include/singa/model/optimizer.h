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

#ifndef SINGA_MODEL_OPTIMIZER_H_
#define SINGA_MODEL_OPTIMIZER_H_

#include <string>
#include <unordered_map>
#include <vector>

#include "singa/core/tensor.h"
#include "singa/proto/model.pb.h"

using std::string;
using std::vector;
using std::unordered_map;
namespace singa {
class Constraint;
class Regularizer;
/// The base class for gradient descent algorithms used to update the model
/// parameters in order to optimize the objective (loss) function.
/// It updates parameters based on the gradients of the loss w.r.t each
/// parameter. Most sub-classes uses first order gradients.
/// An overview of gradient descent algorithms,
/// http://sebastianruder.com/optimizing-gradient-descent/
class Optimizer {
 public:
  Optimizer() = default;
  virtual ~Optimizer();
  /// Setup the optimzier using configurations from serialized string (for
  /// binding languages).
  void Setup(const string& str) {
    OptimizerConf conf;
    conf.ParseFromString(str);
    this->Setup(conf);
  }

  /// Setup the meta fields of the optimizer
  virtual void Setup(const OptimizerConf& conf);
  /// Register the parameter, e.g., create Constraint and Regularizers.
  /// If there is no constraint or regularizer, then no need to register the
  /// parameter.
  virtual void Register(const string& name, const ParamSpec& specs);

  /// Apply the updating algorithm.
  /// No learning rate scaling, gradient constraints/regularization will be
  /// conducted. It assumes all these operations are done either by users or
  /// by Apply(int, const string&, Tensor*, Tensor*).
  /// All sub-classes should override this function.
  virtual void Apply(int step, float lr, const string& name, const Tensor& grad,
                     Tensor* value) = 0;

  /// Apply the updating algorithm.
  /// It will apply regularization and constraint to the parameters if
  /// configured during Register(). If will also scale the learning rate if
  /// configured in ParamSpecs (see Register).
  void Apply(int step, const string& name, Tensor* grad, Tensor* value);

  /// The argument is a function that returns the learning rate given the
  /// current step (i.e., curren running iteration).
  void SetLearningRateGenerator(function<float(int)> func) {
    learning_rate_generator_ = func;
  }
  float GetLearningRate(int step) {
    if (learning_rate_generator_)
      return learning_rate_generator_(step);
    else
      return 0;
  }

 protected:
  function<float(int)> learning_rate_generator_;
  std::unordered_map<std::string, float> learning_rate_multplier_;
  std::unordered_map<std::string, float> weight_decay_multplier_;
  std::unordered_map<std::string, Constraint*> constraints_;
  std::unordered_map<std::string, Regularizer*> regularizers_;
  Constraint* constraint_ = nullptr;
  Regularizer* regularizer_ = nullptr;
};

/// Apply constraints for parameters (gradient).
/// E.g., restrict the norm of parmeter gradients to be within a threshold.
/// \ref http://keras.io/constraints/
/// TODO(wangwei) implement a sub-class for each type of constraint
class Constraint {
 public:
  Constraint() = default;
  explicit Constraint(const ConstraintConf& conf) { Setup(conf); }
  Constraint(const string& type, float threshold)
      : type_(type), threshold_(threshold) {}
  void Setup(const ConstraintConf& conf);
  void Setup(const string& conf_str) {
    ConstraintConf conf;
    conf.ParseFromString(conf_str);
    Setup(conf);
  }
  /// Apply the constraint to a single parmeter object, e.g., W, or b
  /// e.g., clip each gradient if it is too large w.r.t the threshold,
  /// \ref
  /// https://www.reddit.com/r/MachineLearning/comments/31b6x8/gradient_clipping_rnns/
  void Apply(int step, Tensor* grad, Tensor* value);
  /// Apply the constraint for multiple parameter objects together.
  /// \ref https://github.com/Lasagne/Lasagne/blob/master/lasagne/updates.py
  void Apply(int step, const vector<Tensor*>& grads,
             const vector<Tensor*>& values);

 private:
  /// currently only support "L2" norm constraint, i.e., the norm should be less
  /// than the configured threshold_, otherwise, the parameters would be clipped
  /// to make the norm within that threshold.
  /// TODO(wangwei) consider other constraint, e.g., hard clip and unitnorm.
  string type_ = "Unknown";
  float threshold_;
};

inline std::shared_ptr<Constraint> CreateConstraint(std::string type) {
  return std::make_shared<Constraint>();
}
/// Apply regularization for parameters (gradient), e.g., L1 norm and L2 norm.
/// TODO(wangwei) implement a sub-class for each type of regularizer
class Regularizer {
 public:
  Regularizer() = default;
  explicit Regularizer(const RegularizerConf& conf) { Setup(conf); }
  Regularizer(const string& type, float coefficient)
      : type_(type), coefficient_(coefficient) {}
  void Setup(const RegularizerConf& conf);
  void Setup(const string& conf_str) {
    RegularizerConf conf;
    conf.ParseFromString(conf_str);
    Setup(conf);
  }

  /// Apply the regularizer to a single parmeter object, e.g., W, or b
  /// e.g., clip each gradient if it is too large w.r.t the threshold,
  /// \ref
  /// https://www.reddit.com/r/MachineLearning/comments/31b6x8/gradient_clipping_rnns/
  void Apply(int step, Tensor* grad, Tensor* value, float scale = 1.0f);
  /// Apply the regularizer for multiple parameter objects together.
  /// \ref https://github.com/Lasagne/Lasagne/blob/master/lasagne/updates.py
  void Apply(int step, const vector<Tensor*>& grads,
             const vector<Tensor*>& values);

 private:
  /// currently only support "L2" regularizer. type_ is case insensitive.
  /// TODO(wangwei) add more regularizer, e.g., L1.
  string type_ = "NotSet";
  float coefficient_;
};
inline std::shared_ptr<Regularizer> CreateRegularizer(std::string type) {
  return std::make_shared<Regularizer>();
}



// =============Vallina SGD with Momentum=====================================
class SGD : public Optimizer {
 public:
  void Setup(const OptimizerConf& conf);
  /// Apply the updating algorithm.
  void Apply(int step, float lr, const string& name, const Tensor& grad,
             Tensor* value) override;

  /// The argument function returns the momentum value given the current running
  /// step (i.e., iterations/mini-batches).
  void SetMomentumGenerator(std::function<float(int)> func) {
    momentum_generator_ = func;
  }

 private:
  std::unordered_map<string, Tensor> history_gradient_;
  std::function<float(int)> momentum_generator_;
};

// =============Nesterov======================================================
class Nesterov : public Optimizer {
 public:
  void Setup(const OptimizerConf& conf);
  /// Apply the updating algorithm.
  void Apply(int step, float lr, const string& name, const Tensor& grad,
             Tensor* value) override;

  /// The argument function returns the momentum value given the current running
  /// step (i.e., iterations/mini-batches).
  void SetMomentumGenerator(std::function<float(int)> func) {
    momentum_generator_ = func;
  }

 private:
  std::unordered_map<string, Tensor> history_gradient_;
  std::function<float(int)> momentum_generator_;
};

// =============Adagrad=======================================================
class AdaGrad : public Optimizer {
 public:
  void Setup(const OptimizerConf& conf);
  /// Apply the updating algorithm.
  void Apply(int step, float lr, const string& name, const Tensor& grad,
             Tensor* value) override;

 private:
  std::unordered_map<string, Tensor> history_gradient_;
  float delta_;
};
// =============RMSProp=======================================================
class RMSProp : public Optimizer {
 public:
  void Setup(const OptimizerConf& conf);
  /// Apply the updating algorithm.
  void Apply(int step, float lr, const string& name, const Tensor& grad,
             Tensor* value) override;
  virtual ~RMSProp() = default;

 private:
  std::unordered_map<string, Tensor> history_gradient_;
  float delta_, rho_;
};


inline std::shared_ptr<Optimizer> CreateOptimizer(const string& type) {
  std::shared_ptr<Optimizer>  opt;
  if (type == "SGD")
    opt = std::shared_ptr<Optimizer>(new SGD());
  else if (type == "RMSProp")
    opt = std::shared_ptr<Optimizer>(new RMSProp());
  else if (type == "AdaGrad")
    opt = std::shared_ptr<Optimizer>(new AdaGrad());
  else if (type == "Nesterov")
    opt = std::shared_ptr<Optimizer>(new Nesterov());
  else
    LOG(FATAL) << "Unknown optimizer type : " << type;
  return opt;
}
// ============LocalAllReduce for single node multiple workers ==============
/// Updater for training models on a single node with multiple devices (workers)
/// All model parameters are partitioned such that each parameter is updated on
/// one device. In specific, each worker has a model replica. All workers share
/// the same LocalAllReduce instance. Parameters are registered at first, and
/// then after every iteration, the gradients are aggregated by one worker (or
/// device) for parameter updating.
/*
class LocalAllReduce : public Optimizer{
 pulbic:
  LocalAllReduce(Optimizer* opt);
  void Setup(const string& str) {
    AllReduce conf;
    conf.ParseFromString(str);
    this->Setup(conf);
  }
  void Setup(const AllReduce& conf) {}

  /// Register all model parameters.
  /// Instructions include:
  /// 1. Copy parameters from the master worker (who initialized the parameters)
  /// to others.
  /// 2. Partition parameters onto worker devices. For example, model parameter
  /// set is {A, B, C}, nb_workers = 3, then worker 0/1/2 would be in charge of
  /// updating A/B/C respectively. A gradient Tensor for A/B/C would be created
  /// on device 0/1/2, dentoed as GA/GB/GC. 0/1/2 would call the internal opt to
register the specs
  /// for A/B/C.
  void Register(const vector<string>& names,
                const vector<Tensor>& values,
                const vector<ParamSpecs>& specs) override;

  /// Aggregate parameter gradients and call internal opt to do the update.
  /// Continue with the example for Register(), worker 0 would copy B's gradient
  /// to device 1 and add it with GB.  A callback func is added to
  /// 1. check UpdateNow() and call opt to do the real update.
  /// 2. broadcast the new parameters back to worker 0 and 2.
  void Update(int step, float lr, const string& name, const Tensor& grad,
              Tensor* param) override;

  /// Decide when to call the internal Optimizer for real update.
  /// One simple implementation would return true until all workers has
  /// aggregated their gradients. We can also add a user configuration field
  /// to control this, e.g., if do it when 80% workers has aggregated.
  boo UpdateNow();

 private:
  int nb_workers_;
  vector<Tensor> aggregated_gradients_;
};
*/
}
#endif  // SINGA_MODEL_OPTIMIZER_H_
