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
""" Python wrappers for optimizers implemented by C++."""

from . import singa_wrap as singa
import tensor
from proto import model_pb2


class Optimizer(object):
    """Base python optimizer.

    Usages:
        1. construct the optimizer
        2. (optional) register each parameter with its specs.
        3. use the optimizer to update parameter values given parameter
            gradients and other optional info
    """
    def __init__(self, lr=None, momentum=None, decay=None, lr_gen=None,
                 momentum_gen=None, regularizer=None, constraint=None):
        """Constructor.

        Args:
            lr: a constant or a function that generates learning rate given a
                step, which is mutually exclusive with 'lr_gen'.
            momentum: a constant or a function that generates the momentum value
                given a step.
            decay (float): the coefficent for L2 regularizer, which is mutually
                exclusive with 'regularizer'.
            lr_gen (function): a function returns the learning rate given
                the current training step. It is mutually exclusive with lr. If
                both are not set, the apply_with_lr function should be used for
                param updating.
            momentum_gen (function): a function returns the momentum value given
                the current training step. It is mutually exclusive with
                momentum.
            regularizer: an instance of Regularizer or RegularizerConf; If set,
                regularization would be applied in apply_with_lr().
                Users can also do regularization outside.
            constraint: an instance of Constraint or ConstraintConf; If set,
                constraint would be applied inside apply_with_lr(). Users can
                also do regularization outside.
        """
        if lr is not None:
            assert lr_gen is None, 'Cannot set lr and lr_gen at the same time'

            def lr_gen(step):
                return lr
        self.lr_gen = lr_gen
        if momentum is not None:
            assert momentum_gen is None, 'Cannot set momentum and momentum_gen'\
                ' at the same time'

            def momentum_gen(step):
                return momentum
        self.momentum_gen = momentum_gen
        if decay is not None:
            assert regularizer is None, \
                'Cannot set decay and regularizer at the same time'
            regularizer = L2Regularizer(decay)
        if regularizer is not None:
            if type(regularizer) is model_pb2.RegularizerConf:
                self.regularizer = CppRegularizer(regularizer)
            else:
                self.regularizer = regularizer
        else:
            self.regularizer = None
        if constraint is not None:
            if type(constraint) is model_pb2.ConstraintConf:
                self.constraint = CppConstraint(constraint)
            else:
                self.constraint = constraint
        else:
            self.constraint = None
        self.regularizers = {}
        self.constraints = {}

    def register(self, name, specs):
        """Register the param specs, including creating regularizer and
        constraint per param object. Param specific regularizer and constraint
        have higher priority than the global ones.

        Args:
            name (str): parameter name
            specs (ParamSpec): protobuf obj
        """
        if specs.has_regularizer():
            self.regularizers[name] = CppRegularizer(specs.constraint)
        if specs.has_constraint():
            self.constraints[name] = CppConstraint(specs.regularizer)
        if specs.has_lr_mult():
            self.learning_rate_multiplier[name] = specs.lr_mult()
        if specs.has_decay_mult():
            self.decay_multiplier[name] = specs.decay_mult()

    def apply_regularizer_constraint(self, value, grad, name=None, step=None):
        """Apply regularization and constraint if available.

        If there are both global regularizer (constraint) and param specific
        regularizer (constraint), it would use the param specific one.

        Args:
            value (Tensor): parameter value Tensor
            grad (Tensor): parameter gradient Tensor
            name (string): to get parameter specific regularizer or constraint
            step (int): some regularizer or constraint would use step

        Return:
            the updated gradient Tensor
        """
        if name is not None and name in self.constraints:
            self.constraints[name].apply(value, grad, step)
        elif self.constraint is not None:
            self.constraint.apply(step, value, grad)

        if name is not None and name in self.regularizers:
            self.regularizers[name].apply(value, grad, step)
        elif self.regularizer is not None:
            self.regularizer.apply(step, value, grad)
        return grad

    def apply_with_lr(self, step, lr, grad, value, name=None):
        """Do update with given learning rate.

        The subclass optimizer must override this function.
        Args:
            step (int): training step (could be iteration or epoch)
            lr (float): learning rate
            grad (Tensor): parameter gradient
            value (Tesnor): parameter value
            name (string): paramter name to retrieval parameter specific
                updating rules (including regularizer and constraint)

        Return:
            updated parameter value
        """
        assert False, 'This is the base function, pls call the subclass func'
        return value

    def apply(self, step, grad, value, name=None):
        """Do update assume the learning rate generator is set.

        The subclass optimizer does not need to override this function.
        Args:
            step (int): training step (could be iteration or epoch)
            grad (Tensor): parameter gradient
            value (Tesnor): parameter value
            name (string): paramter name to retrieval parameter specific
                updating rules (including regularizer and constraint)

        Return:
            updated parameter value
        """

        assert self.lr_gen is not None, 'Learning rate generator is not set.'\
            'Either set the lr_gen in constructor or call apply_with_lr'
        lr = self.lr_gen(step)
        return self.apply_with_lr(step, lr, grad, value, name)


class SGD(Optimizer):
    def __init__(self, lr=None, momentum=None, decay=None, **kwargs):
        """The vallina Stochasitc Gradient Descent algorithm.

        See the base Optimizer for all arguments.
        """
        super(SGD, self).__init__(lr, momentum, decay)
        conf = model_pb2.OptimizerConf()
        self.opt = singa.CreateOptimizer('SGD')
        self.opt.Setup(conf.SerializeToString())

    def apply_with_lr(self, step, lr, grad, value, name):
        self.apply_regularizer_constraint(step, value, grad, name)
        self.opt.Apply(step, lr, name, grad.singa_tensor, value.singa_tensor)
        return value


class Nesterov(Optimizer):
    def __init__(self, lr=None, momentum=0.9, decay=None, **kwargs):
        """The SGD with Nesterov momentum

        See the base Optimizer for all arguments.
        """
        super(Nesterov, self).__init__(lr, momentum, decay, kwargs)
        conf = model_pb2.OptimizerConf()
        self.opt = singa.CreateOptimizer('Nesterov')
        self.opt.Setup(conf.SerializeToString())

    def apply_with_lr(self, step, lr, grad, value, name):
        self.apply_regularizer_constraint(step, value, grad, name)
        self.opt.Apply(step, lr, name, grad.singa_tensor, value.singa_tensor)
        return value


class AdaGrad(Optimizer):
    def __init__(self, epsilon=1e-8, lr=None, decay=None, **kwargs):
        """AdaGrad optimizer.

        See the base Optimizer for all constructor args.
        Args:
            epsilon (float): small number for preventing numeric error.
        """
        super(RMSProp, self).__init__(lr, decay, **kwargs)
        conf = model_pb2.OptimizerConf()
        conf.delta = epsilon
        self.opt = singa.CreateOptimizer('AdaGrad')
        self.opt.Setup(conf.SerializeToString())

    def apply_with_lr(self, step, lr, grad, value, name):
        grad = self.apply_regularizer_constraint(step, value, grad, name)
        self.opt.Apply(step, lr,  name, grad.singa_tensor, value.singa_tensor)
        return value


class RMSProp(Optimizer):
    def __init__(self, rho=0.9, epsilon=1e-8, lr=None, decay=None, **kwargs):
        """RMSProp optimizer.

        See the base Optimizer for all constructor args.
        Args:
            rho (float): float within [0, 1]
            epsilon (float): small value for preventing numeric error
        """
        super(RMSProp, self).__init__(lr, decay, kwargs)
        conf = model_pb2.OptimizerConf()
        conf.rho = rho
        conf.delta = epsilon
        self.opt = singa.CreateOptimizer('RMSProp')
        self.opt.Setup(conf.SerializeToString())

    def apply_with_lr(self, step, lr, grad, value, name):
        grad = self.apply_regularizer_constraint(step, value, grad, name)
        self.opt.Apply(step, lr,  name, grad.singa_tensor, value.singa_tensor)
        return value


class Regularizer(object):
    """Base Python regularizer for parameter gradients.
    """
    def apply(self, value, grad):
        assert False, 'Not Implemented. Call the subclass function.'
        return grad


class CppRegularizer(Regularizer):
    """Wrapper for regularizer implemented using C++.
    """
    def __init__(self, conf):
        """Constructor.

        Args:
            conf (RegularizerConf): protobuf message for the configuration.
        """
        self.reg = singa.CreateRegularizer(conf.type)
        self.reg.Setup(conf.SerializeToString())

    def apply(self, step, value, grad):
        self.reg.Apply(step, value.singa_tensor, grad.singa_tensor)
        return grad


class L2Regularizer(Regularizer):
    """L2 regularization"""
    def __init__(self, coefficient):
        """
        Args:
            coefficient (float): regularization coefficient.
        """
        self.coefficient = coefficient

    def apply(self, step, value, grad, coefficient=None):
        if coefficient is None:
            assert self.coefficient is not None, 'Must set the coefficient'
            coefficient = self.coefficient
        tensor.axpy(coefficient, value, grad)
        return grad


class Constraint(object):
    """Base Python constraint class for paramter gradients.
    """
    def apply(self, step, value, grad):
        return grad


class CppConstraint(Constraint):
    """Wrapper for constraints implemented using C++.
    """
    def __init__(self, conf):
        """Constructor.

        Args:
            conf (ConstraintConf): protobuf message for the configuration.
        """
        self.constraint = singa.CreateConstraint(conf.type)
        self.constraint.Setup(conf.SerializeToString())

    def apply(self, step, value, grad):
        self.constraint.Apply(step, value.singa_tensor, grad.singa_tensor)
        return grad


class L2Constraint(Constraint):
    """Rescale the gradient to make the L2 norm <= a given threshold.
    """
    def __init__(self, threshold=None):
        self.threshold = threshold

    def apply(self, step, value, grad, threshold=None):
        if threshold is None:
            assert self.threshold is not None, 'Must set the threshold'
            threshold = self.threshold
        nrm = grad.nrm2()
        grad *= threshold / nrm
        return grad
