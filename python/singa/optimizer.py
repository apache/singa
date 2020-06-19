# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# =============================================================================
'''This module includes a set of optimizers for updating model parameters.

Note: This module is deprecated. Please use the opt module.

Example usage::

  from singa import optimizer
  from singa import tensor

  sgd = optimizer.SGD(lr=0.01, momentum=0.9, weight_decay=1e-4)
  p = tensor.Tensor((3,5))
  p.uniform(-1, 1)
  g = tensor.Tensor((3,5))
  g.gaussian(0, 0.01)

  sgd.apply(1, g, p, 'param')  # use the global lr=0.1 for epoch 1
  sgd.apply_with_lr(2, 0.03, g, p, 'param')  # use lr=0.03 for epoch 2
'''
from __future__ import division
from __future__ import absolute_import

from builtins import object
import math

from . import singa_wrap as singa
from . import tensor
from .proto import model_pb2


class Optimizer(object):
    '''The base python optimizer class.

    Typically, an optimizer is used as follows:

    1. construct the optimizer
    2. (optional) register each parameter with its specs.
    3. use the optimizer to update parameter values given parameter gradients
       and other optional info

    The subclasses should override the apply_with_lr function to do the real
    parameter udpate.

    Args:
        lr (float): a constant value for the learning rate
        momentum (float): a constant value for the momentum value
        weight_decay (float): the coefficent for L2 regularizer, which is
            mutually exclusive with 'regularizer'.
        regularizer: an instance of Regularizer or RegularizerConf; If set,
            regularization would be applied in apply_with_lr().
            Users can also do regularization outside.
        constraint: an instance of Constraint or ConstraintConf; If set,
            constraint would be applied inside apply_with_lr(). Users can
            also apply constraint outside.
    '''

    def __init__(self,
                 lr=None,
                 momentum=None,
                 weight_decay=None,
                 regularizer=None,
                 constraint=None):
        self.lr = lr
        self.momentum = momentum
        if weight_decay is not None:
            assert regularizer is None, \
                'Cannot set weight_decay and regularizer at the same time'
            regularizer = L2Regularizer(weight_decay)

        if regularizer is not None:
            if isinstance(regularizer, model_pb2.RegularizerConf):
                self.regularizer = CppRegularizer(regularizer)
            else:
                self.regularizer = regularizer
        else:
            self.regularizer = None
        if constraint is not None:
            if isinstance(constraint, model_pb2.ConstraintConf):
                self.constraint = CppConstraint(constraint)
            else:
                self.constraint = constraint
        else:
            self.constraint = None
        self.regularizers = {}
        self.constraints = {}
        self.decay_multiplier = {}
        self.learning_rate_multiplier = {}

    def register(self, name, specs):
        '''Register the param specs, including creating regularizer and
        constraint per param object. Param specific regularizer and constraint
        have higher priority than the global ones. If all parameters share the
        same setting for learning rate, regularizer and constraint, then there
        is no need to call this function.

        Args:
            name (str): parameter name
            specs (ParamSpec): protobuf obj, including regularizer and
                constraint, multipliers for learning rate and weight decay.
        '''
        assert isinstance(specs, model_pb2.ParamSpec), \
            'specs should be model_pb2.ParamSpec instance'
        if specs.HasField('regularizer'):
            self.regularizers[name] = CppRegularizer(specs.regularizer)
        elif specs.decay_mult != 1:
            self.regularizers[name] = L2Regularizer(
                specs.decay_mult * self.regularizer.coefficient)

        if specs.HasField('constraint'):
            self.constraints[name] = CppConstraint(specs.constraint)

        if specs.lr_mult != 1:
            self.learning_rate_multiplier[name] = specs.lr_mult

    def apply_regularizer_constraint(self,
                                     epoch,
                                     value,
                                     grad,
                                     name=None,
                                     step=-1):
        '''Apply regularization and constraint if available.

        If there are both global regularizer (constraint) and param specific
        regularizer (constraint), it would use the param specific one.

        Args:
            epoch (int): training epoch ID
            value (Tensor): parameter value Tensor
            grad (Tensor): parameter gradient Tensor
            name (string): to get parameter specific regularizer or constraint
            step (int): iteration ID within one epoch

        Returns:
            the updated gradient Tensor
        '''
        if name is not None and name in self.constraints:
            grad = self.constraints[name].apply(epoch, value, grad, step)
        elif self.constraint is not None:
            grad = self.constraint.apply(epoch, value, grad, step)

        if name is not None and name in self.regularizers:
            grad = self.regularizers[name].apply(epoch, value, grad, step)
        elif self.regularizer is not None:
            grad = self.regularizer.apply(epoch, value, grad, step)
        return grad

    def apply_with_lr(self, epoch, lr, grad, value, name=None, step=-1):
        '''Do update of parameters with given learning rate if the grad is not
        empty.

        The subclass optimizer must override this function.
        This function do nothing if the grad is empty.

        Args:
            epoch (int): training epoch ID
            lr (float): learning rate
            grad (Tensor): parameter gradient
            value (Tesnor): parameter value
            name (string): paramter name to index parameter specific
                updating rules (including regularizer and constraint)
            step (int): iteration ID within one epoch

        Returns:
            updated parameter value
        '''
        assert False, 'This is the base function, pls call the subclass func'

    def apply(self, epoch, grad, value, name=None, step=-1):
        '''Do update assuming the learning rate generator is set.

        The subclass optimizer does not need to override this function.

        Args:
            epoch (int): training epoch ID
            grad (Tensor): parameter gradient
            value (Tesnor): parameter value
            name (string): paramter name to retrieval parameter specific
                updating rules (including regularizer and constraint)
            step (int): training iteration ID within one epoch

        Return:
            updated parameter value
        '''
        assert self.lr is not None, 'Must set the learning rate, i.e. "lr"'
        return self.apply_with_lr(epoch, self.lr, grad, value, name, step)


class SGD(Optimizer):
    '''The vallina Stochasitc Gradient Descent algorithm with momentum.

    See the base Optimizer for all arguments.
    '''

    def __init__(self,
                 lr=None,
                 momentum=None,
                 weight_decay=None,
                 regularizer=None,
                 constraint=None):
        super(SGD, self).__init__(lr, momentum, weight_decay, regularizer,
                                  constraint)
        conf = model_pb2.OptimizerConf()
        if self.momentum is not None:
            conf.momentum = self.momentum
        conf.type = 'sgd'
        self.opt = singa.CreateOptimizer('SGD'.encode())
        self.opt.Setup(conf.SerializeToString())

    def apply_with_lr(self, epoch, lr, grad, value, name, step=-1):
        if grad.is_empty():
            return value
        grad = self.apply_regularizer_constraint(epoch, value, grad, name, step)
        if name is not None and name in self.learning_rate_multiplier:
            lr = lr * self.learning_rate_multiplier[name]
        self.opt.Apply(epoch, lr, name.encode(), grad.data, value.data)
        return value


class Nesterov(Optimizer):
    '''The SGD with Nesterov momentum.

    See the base Optimizer for all arguments.
    '''

    def __init__(self,
                 lr=None,
                 momentum=0.9,
                 weight_decay=None,
                 regularizer=None,
                 constraint=None):
        super(Nesterov, self).__init__(lr, momentum, weight_decay, regularizer,
                                       constraint)
        conf = model_pb2.OptimizerConf()
        if self.momentum is not None:
            conf.momentum = momentum
        conf.type = 'nesterov'
        self.opt = singa.CreateOptimizer('Nesterov'.encode())
        self.opt.Setup(conf.SerializeToString())

    def apply_with_lr(self, epoch, lr, grad, value, name, step=-1):
        if grad.is_empty():
            return value

        grad = self.apply_regularizer_constraint(epoch, value, grad, name, step)
        if name is not None and name in self.learning_rate_multiplier:
            lr = lr * self.learning_rate_multiplier[name]
        self.opt.Apply(epoch, lr, name.encode(), grad.data, value.data)
        return value


class RMSProp(Optimizer):
    '''RMSProp optimizer.

    See the base Optimizer for all constructor args.

    Args:
        rho (float): float within [0, 1]
        epsilon (float): small value for preventing numeric error
    '''

    def __init__(self,
                 rho=0.9,
                 epsilon=1e-8,
                 lr=None,
                 weight_decay=None,
                 regularizer=None,
                 constraint=None):
        super(RMSProp, self).__init__(lr, None, weight_decay, regularizer,
                                      constraint)
        conf = model_pb2.OptimizerConf()
        conf.rho = rho
        conf.delta = epsilon
        self.opt = singa.CreateOptimizer('RMSProp'.encode())
        self.opt.Setup(conf.SerializeToString())

    def apply_with_lr(self, epoch, lr, grad, value, name, step=-1):
        if grad.is_empty():
            return value

        grad = self.apply_regularizer_constraint(epoch, value, grad, name, step)
        if name is not None and name in self.learning_rate_multiplier:
            lr = lr * self.learning_rate_multiplier[name]
        self.opt.Apply(step, lr, name.encode(), grad.data, value.data)
        return value


class AdaGrad(Optimizer):
    '''AdaGrad optimizer.

    See the base Optimizer for all constructor args.

    Args:
        epsilon (float): small number for preventing numeric error.
    '''

    def __init__(self,
                 epsilon=1e-8,
                 lr=None,
                 weight_decay=None,
                 lr_gen=None,
                 regularizer=None,
                 constraint=None):
        super(AdaGrad, self).__init__(lr, None, weight_decay, regularizer,
                                      constraint)
        conf = model_pb2.OptimizerConf()
        conf.delta = epsilon
        conf.type = 'adagrad'
        self.opt = singa.CreateOptimizer('AdaGrad'.encode())
        self.opt.Setup(conf.SerializeToString())

    def apply_with_lr(self, epoch, lr, grad, value, name, step=-1):
        if grad.is_empty():
            return value

        grad = self.apply_regularizer_constraint(epoch, value, grad, name, step)
        if name is not None and name in self.learning_rate_multiplier:
            lr = lr * self.learning_rate_multiplier[name]
        self.opt.Apply(epoch, lr, name.encode(), grad.data, value.data)
        return value


class Adam(Optimizer):
    '''Adam optimizer.

    See the base Optimizer for all constructor args.

    Args:
        beta_1(float): coefficient of momentum
        beta_2(float): coefficient of aggregated squared gradient
        epsilon (float): small value for preventing numeric error
    '''

    def __init__(self,
                 beta_1=0.9,
                 beta_2=0.999,
                 epsilon=1e-8,
                 lr=None,
                 weight_decay=None,
                 regularizer=None,
                 constraint=None):
        super(Adam, self).__init__(lr, None, weight_decay, regularizer,
                                   constraint)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0
        self.last_epoch = -1
        self.last_step = -1

    def apply_with_lr(self, epoch, lr, grad, value, name, step):
        '''Update one parameter object.

        Args:
            step(int): the accumulated training iterations, not the iteration ID
        '''
        if grad.is_empty():
            return value

        assert step != -1, 'step should >= 0'
        if epoch != self.last_epoch or step != self.last_step:
            self.t += 1
            self.last_step = step
            self.last_epoch = epoch
        grad = self.apply_regularizer_constraint(epoch, value, grad, name, step)
        if name is not None and name in self.learning_rate_multiplier:
            lr = lr * self.learning_rate_multiplier[name]
        if name not in self.m or name not in self.v:
            self.m[name] = tensor.Tensor(grad.shape, grad.device, grad.dtype)
            self.m[name].set_value(0)
            self.v[name] = tensor.Tensor(grad.shape, grad.device, grad.dtype)
            self.v[name].set_value(0)

        self.m[name] *= self.beta_1
        tensor.axpy(1 - self.beta_1, grad, self.m[name])
        self.v[name] *= self.beta_2
        tensor.axpy(1 - self.beta_2, tensor.square(grad), self.v[name])
        alpha = lr * math.sqrt(1 - math.pow(self.beta_2, self.t)) \
            / (1 - math.pow(self.beta_1, self.t))
        value -= alpha * self.m[name] / (tensor.sqrt(self.v[name]) +
                                         self.epsilon)
        return value


class Regularizer(object):
    '''Base Python regularizer for parameter gradients.'''

    def apply(self, epoch, value, grad, step=-1):
        assert False, 'Not Implemented. Call the subclass function.'


class CppRegularizer(Regularizer):
    '''Wrapper for regularizer implemented using C++.

    Args:
        conf (RegularizerConf): protobuf message for the configuration.
    '''

    def __init__(self, conf):
        self.reg = singa.CreateRegularizer(conf.type)
        self.reg.Setup(conf.SerializeToString())

    def apply(self, epoch, value, grad, step=-1):
        self.reg.Apply(epoch, value.data, grad.data)
        return grad


class L2Regularizer(Regularizer):
    '''L2 regularization

    Args:
        coefficient (float): regularization coefficient.
    '''

    def __init__(self, coefficient):
        self.coefficient = coefficient

    def apply(self, epoch, value, grad, step=-1):
        # print coefficient, value.l1(), grad.l1()
        if self.coefficient != 0:
            tensor.axpy(self.coefficient, value, grad)
        return grad


class Constraint(object):
    '''Base Python constraint class for paramter gradients'''

    def apply(self, epoch, value, grad, step=-1):
        return grad


class CppConstraint(Constraint):
    '''Wrapper for constraints implemented using C++.

    Args:
        conf (ConstraintConf): protobuf message for the configuration.
    '''

    def __init__(self, conf):
        self.constraint = singa.CreateConstraint(conf.type)
        self.constraint.Setup(conf.SerializeToString())

    def apply(self, epoch, value, grad, step=-1):
        self.constraint.Apply(epoch, value.data, grad.data, step)
        return grad


class L2Constraint(Constraint):
    '''Rescale the gradient to make the L2 norm <= a given threshold'''

    def __init__(self, threshold=None):
        self.threshold = threshold

    def apply(self, epoch, value, grad, step=-1):
        nrm = grad.l2()
        grad *= self.threshold / nrm
        return grad
