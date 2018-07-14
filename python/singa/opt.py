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

'''This module includes a set of optimizers for updating model parameters.
It replaces the old optimizers from optimizer.py'''

from singa import tensor


class Optimizer(object):
    r"""Base optimizer.

    Args:
        config (Dict): specify the default values of configurable variables.
    """

    def __init__(self, config):
        self.default_config = config
        self.step = 0
        self.param2config = {}
        self.param2state = {}

    def update(self, param, grad):
        r"""Update the param values with given gradients.

        Args:
            param(Tensor): param values to be updated in-place
            grad(Tensor): param gradients; the values may be updated
                    in this function; do not use it anymore
        """
        pass

    def step(self):
        r"""To increment the step counter"""
        self.step += 1

    def register(self, param_group, config):
        for param in param_group:
            assert param not in self.param2config, 'param is already registered'

            self.param2config[param] = config

    def load(self):
        pass

    def save(self):
        pass


class SGD(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        lr(float): learning rate
        momentum(float, optional): momentum factor(default: 0)
        weight_decay(float, optional): weight decay(L2 penalty)(default: 0)
        dampening(float, optional): dampening for momentum(default: 0)
        nesterov(bool, optional): enables Nesterov momentum(default: False)

    Example:
        >> > from singa import opt
        >> > optimizer = opt.SGD(lr=0.1, momentum=0.9)
        >> > optimizer.update()

    __ http: // www.cs.toronto.edu / %7Ehinton / absps / momentum.pdf

    .. note::
        The implementation of SGD with Momentum / Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and: math: `\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, lr=0.1, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")
        super(SGD, self).__init__(defaults)

    def update(self, param, grad):
        """Performs a single optimization step.

        Arguments:
                param(Tensor): param values to be update in-place
                grad(Tensor): param gradients; the values may be updated
                        in this function; cannot use it anymore
        """
        group = self.default_config
        if param in self.param2config:
            group = self.param2config[param]
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']

        if weight_decay != 0:
            grad += param * weight_decay
        if momentum != 0:
            if param not in self.param2state:
                self.param2state[param] = {}
            param_state = self.param2state[param]
            if 'momentum_buffer' not in param_state:
                buf = param_state[
                    'momentum_buffer'] = tensor.zeros_like(param)
                buf *= momentum
                buf += grad
            else:
                buf = param_state['momentum_buffer']
                buf *= momentum
                buf += (1 - dampening) * grad
            if nesterov:
                grad += momentum * buf
            else:
                grad = buf
        param -= grad * group['lr']
