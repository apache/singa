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

from . import singa_wrap as singa
import tensor
from proto import model_pb2


class Optimizer(object):
    '''The base python optimizer class.

    Typically, an optimizer is used as follows:

    1. construct the optimizer
    2. (optional) register each parameter with its specs.
    3. use the optimizer to update parameter values given parameter
        gradients and other optional info

    The subclasses should override the apply_with_lr function to do the real
    parameter udpate.

    Args:
        lr (float): a constant for the learning rate, mutually exclusive with
            'lr_gen'.
        momentum (float): a constant for the momentum value
        decay (float): the coefficent for L2 regularizer, which is mutually
            exclusive with 'regularizer'.
        lr_gen (function): a function returns the learning rate given
            the current training step/epoch. It is mutually exclusive with lr.
            If both are not set, the apply_with_lr function should be used for
            param updating.
        regularizer: an instance of Regularizer or RegularizerConf; If set,
            regularization would be applied in apply_with_lr().
            Users can also do regularization outside.
        constraint: an instance of Constraint or ConstraintConf; If set,
            constraint would be applied inside apply_with_lr(). Users can
            also do regularization outside.
    '''
    def __init__(self, lr=None, momentum=None, decay=None, lr_gen=None,
                 regularizer=None, constraint=None):
        if lr is not None:
            assert lr_gen is None, 'Cannot set lr and lr_gen at the same time'

            def lr_gen(epoch):
                return lr
        self.lr_gen = lr_gen
        self.momentum = momentum
        if decay is not None:
            assert regularizer is None, \
                'Cannot set decay and regularizer at the same time'
            regularizer = L2Regularizer(decay)
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
        have higher priority than the global ones.

        Args:
            name (str): parameter name
            specs (ParamSpec): protobuf obj, including regularizer and
                constraint, multipliers for learning rate and weight decay.

        '''
	assert type(specs) == model_pb2.ParamSpec, \
		'specs should be model_pb2.ParamSpec instance'
        if specs.HasField('regularizer'):
            self.regularizers[name] = CppRegularizer(specs.regularizer)
        if specs.HasField('constraint'):
            self.constraints[name] = CppConstraint(specs.constraint)
        if specs.lr_mult != 1:
            self.learning_rate_multiplier[name] = specs.lr_mult
        if specs.decay_mult != 1:
            self.decay_multiplier[name] = specs.decay_mult

    def apply_regularizer_constraint(self, value, grad, name=None, epoch=None):
        '''Apply regularization and constraint if available.

        If there are both global regularizer (constraint) and param specific
        regularizer (constraint), it would use the param specific one.

        Args:
            value (Tensor): parameter value Tensor
            grad (Tensor): parameter gradient Tensor
            name (string): to get parameter specific regularizer or constraint
            epoch (int): some regularizer or constraint would use epoch

        Returns:
            the updated gradient Tensor
        '''
        if name is not None and name in self.constraints:
            self.constraints[name].apply(value, grad, epoch)
        elif self.constraint is not None:
            self.constraint.apply(epoch, value, grad)

        if name is not None and name in self.regularizers:
            self.regularizers[name].apply(value, grad, epoch)
        elif self.regularizer is not None:
            self.regularizer.apply(epoch, value, grad)
        return grad

    def apply_with_lr(self, epoch, lr, grad, value, name=None):
        '''Do update with given learning rate.

        The subclass optimizer must override this function.

        Args:
            epoch (int): training epoch (could be iteration or epoch)
            lr (float): learning rate
            grad (Tensor): parameter gradient
            value (Tesnor): parameter value
            name (string): paramter name to retrieval parameter specific
                updating rules (including regularizer and constraint)

        Returns:
            updated parameter value
        '''
        assert False, 'This is the base function, pls call the subclass func'
        return value

    def apply(self, epoch, grad, value, name=None):
        '''Do update assuming the learning rate generator is set.

        The subclass optimizer does not need to override this function.

        Args:
            epoch (int): training epoch (could be iteration or epoch)
            grad (Tensor): parameter gradient
            value (Tesnor): parameter value
            name (string): paramter name to retrieval parameter specific
                updating rules (including regularizer and constraint)

        Return:
            updated parameter value
        '''
        assert self.lr_gen is not None, 'Learning rate generator is not set.'\
            'Either set the lr_gen in constructor or call apply_with_lr'
        lr = self.lr_gen(epoch)
        return self.apply_with_lr(epoch, lr, grad, value, name)


class SGD(Optimizer):
    '''The vallina Stochasitc Gradient Descent algorithm with momentum.

    See the base Optimizer for all arguments.
    '''

    def __init__(self, lr=None, momentum=None, decay=None, lr_gen=None,
                 regularizer=None, constraint=None):
        super(SGD, self).__init__(lr, momentum, decay, lr_gen, regularizer,
                                  constraint)
        conf = model_pb2.OptimizerConf()
        conf.momentum = self.momentum
        conf.type = 'sgd'
        self.opt = singa.CreateOptimizer('SGD')
        self.opt.Setup(conf.SerializeToString())

    def apply_with_lr(self, epoch, lr, grad, value, name):
        self.apply_regularizer_constraint(epoch, value, grad, name)
        self.opt.Apply(epoch, lr, name, grad.singa_tensor, value.singa_tensor)
        return value


class Nesterov(Optimizer):
    '''The SGD with Nesterov momentum.

    See the base Optimizer for all arguments.
    '''

    def __init__(self, lr=None, momentum=0.9, decay=None, lr_gen=None,
                 regularizer=None, constraint=None):
        super(Nesterov, self).__init__(lr, momentum, decay, lr_gen, regularizer,
                                       constraint)
        conf = model_pb2.OptimizerConf()
        conf.momentum = momentum
        conf.type = 'nesterov'
        self.opt = singa.CreateOptimizer('Nesterov')
        self.opt.Setup(conf.SerializeToString())

    def apply_with_lr(self, epoch, lr, grad, value, name):
        self.apply_regularizer_constraint(epoch, value, grad, name)
        self.opt.Apply(epoch, lr, name, grad.singa_tensor, value.singa_tensor)
        return value


class AdaGrad(Optimizer):
    '''AdaGrad optimizer.

    See the base Optimizer for all constructor args.

    Args:
        epsilon (float): small number for preventing numeric error.
    '''
    def __init__(self, epsilon=1e-8, lr=None, decay=None, lr_gen=None,
                 regularizer=None, constraint=None):
        super(RMSProp, self).__init__(lr, decay, lr_gen, regularizer,
                                      constraint)
        conf = model_pb2.OptimizerConf()
        conf.delta = epsilon
        conf.type = 'adagrad'
        self.opt = singa.CreateOptimizer('AdaGrad')
        self.opt.Setup(conf.SerializeToString())

    def apply_with_lr(self, epoch, lr, grad, value, name):
        grad = self.apply_regularizer_constraint(epoch, value, grad, name)
        self.opt.Apply(epoch, lr,  name, grad.singa_tensor, value.singa_tensor)
        return value


class RMSProp(Optimizer):
    '''RMSProp optimizer.

    See the base Optimizer for all constructor args.

    Args:
        rho (float): float within [0, 1]
        epsilon (float): small value for preventing numeric error
    '''

    def __init__(self, rho=0.9, epsilon=1e-8, lr=None, decay=None, lr_gen=None,
                 regularizer=None, constraint=None):
        super(RMSProp, self).__init__(lr, decay, lr_gen, regularizer,
                                      constraint)
        conf = model_pb2.OptimizerConf()
        conf.rho = rho
        conf.delta = epsilon
        self.opt = singa.CreateOptimizer('RMSProp')
        self.opt.Setup(conf.SerializeToString())

    def apply_with_lr(self, epoch, lr, grad, value, name):
        grad = self.apply_regularizer_constraint(epoch, value, grad, name)
        self.opt.Apply(epoch, lr,  name, grad.singa_tensor, value.singa_tensor)
        return value


class Regularizer(object):
    '''Base Python regularizer for parameter gradients.'''

    def apply(self, value, grad):
        assert False, 'Not Implemented. Call the subclass function.'
        return grad


class CppRegularizer(Regularizer):
    '''Wrapper for regularizer implemented using C++.

    Args:
        conf (RegularizerConf): protobuf message for the configuration.
    '''

    def __init__(self, conf):
        self.reg = singa.CreateRegularizer(conf.type)
        self.reg.Setup(conf.SerializeToString())

    def apply(self, epoch, value, grad):
        self.reg.Apply(epoch, value.singa_tensor, grad.singa_tensor)
        return grad


class L2Regularizer(Regularizer):
    '''L2 regularization

    Args:
        coefficient (float): regularization coefficient.
    '''

    def __init__(self, coefficient):
        self.coefficient = coefficient

    def apply(self, epoch, value, grad, coefficient=None):
        if coefficient is None:
            assert self.coefficient is not None, 'Must set the coefficient'
            coefficient = self.coefficient
        tensor.axpy(coefficient, value, grad)
        return grad


class Constraint(object):
    '''Base Python constraint class for paramter gradients'''

    def apply(self, epoch, value, grad):
        return grad


class CppConstraint(Constraint):
    '''Wrapper for constraints implemented using C++.

    Args:
        conf (ConstraintConf): protobuf message for the configuration.
    '''
    def __init__(self, conf):
        self.constraint = singa.CreateConstraint(conf.type)
        self.constraint.Setup(conf.SerializeToString())

    def apply(self, epoch, value, grad):
        self.constraint.Apply(epoch, value.singa_tensor, grad.singa_tensor)
        return grad


class L2Constraint(Constraint):
    '''Rescale the gradient to make the L2 norm <= a given threshold'''

    def __init__(self, threshold=None):
        self.threshold = threshold

    def apply(self, epoch, value, grad, threshold=None):
        if threshold is None:
            assert self.threshold is not None, 'Must set the threshold'
            threshold = self.threshold
        nrm = grad.l2()
        grad *= threshold / nrm
        return grad
