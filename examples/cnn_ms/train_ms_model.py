#
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
#

from singa import singa_wrap as singa
from singa import device
from singa import tensor
from singa import opt
from singa import autograd
from singa.opt import Optimizer
from singa.opt import DecayScheduler
from singa.opt import Constant
import numpy as np
import time
import argparse
from PIL import Image

np_dtype = {"float16": np.float16, "float32": np.float32}

singa_dtype = {"float16": tensor.float16, "float32": tensor.float32}

### MSOptimizer
class MSOptimizer(Optimizer):
    def __call__(self, loss):
        pn_p_g_list = self.call_with_returns(loss)
        self.step()
        return pn_p_g_list

    def call_with_returns(self, loss):
        # print ("call_with_returns loss.data: \n", loss.data)
        pn_p_g_list = []
        for p, g in autograd.backward(loss):
            if p.name is None:
                p.name = id(p)
            self.apply(p.name, p, g)
            # print ("call with returns")
            # print ("p.name: \n", p.name)
            # print ("p.data: \n", p.data)
            # print ("g.data: \n", g.data)
            pn_p_g_list.append([p.name, p, g])  # need iterables
        return pn_p_g_list

class MSSGD(MSOptimizer):
    """Implements stochastic gradient descent (optionally with momentum).
    Nesterov momentum is based on the formula from `On the importance of initialization and momentum in deep learning`__.
    Args:
        lr(float): learning rate
        momentum(float, optional): momentum factor(default: 0)
        weight_decay(float, optional): weight decay(L2 penalty)(default: 0)
        dampening(float, optional): dampening for momentum(default: 0)
        nesterov(bool, optional): enables Nesterov momentum(default: False)
    Typical usage example:
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

    def __init__(self,
                 lr=0.1,
                 momentum=0,
                 dampening=0,
                 weight_decay=0,
                 nesterov=False,
                 dtype=tensor.float32):
        super(MSSGD, self).__init__(lr, dtype)

        # init momentum
        if type(momentum) == float or type(momentum) == int:
            if momentum < 0.0:
                raise ValueError("Invalid momentum value: {}".format(momentum))
            self.momentum = Constant(momentum)
        elif isinstance(momentum, DecayScheduler):
            self.momentum = momentum
            momentum = momentum.init_value
        else:
            raise TypeError("Wrong momentum type")
        self.mom_value = self.momentum(self.step_counter).as_type(self.dtype)

        # init dampening
        if type(dampening) == float or type(dampening) == int:
            self.dampening = Constant(dampening)
        elif isinstance(dampening, DecayScheduler):
            self.dampening = dampening
            dampening = dampening.init_value
        else:
            raise TypeError("Wrong dampening type")
        self.dam_value = self.dampening(self.step_counter).as_type(self.dtype)

        # init weight_decay
        if type(weight_decay) == float or type(weight_decay) == int:
            if weight_decay < 0.0:
                raise ValueError(
                    "Invalid weight_decay value: {}".format(weight_decay))
            self.weight_decay = Constant(weight_decay)
        elif isinstance(weight_decay, DecayScheduler):
            self.weight_decay = weight_decay
        else:
            raise TypeError("Wrong weight_decay type")
        self.decay_value = self.weight_decay(self.step_counter).as_type(
            self.dtype)

        # init other params
        self.nesterov = nesterov
        self.moments = dict()

        # check value
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a momentum and zero dampening")

    def apply(self, param_name, param_value, param_grad):
        """Performs a single optimization step.
        Args:
                param_name(String): the name of the param
                param_value(Tensor): param values to be update in-place
                grad(Tensor): param gradients; the values may be updated
                        in this function; cannot use it anymore
        """
        assert param_value.shape == param_grad.shape, ("shape mismatch",
                                                       param_value.shape,
                                                       param_grad.shape)
        self.device_check(param_value, self.step_counter, self.lr_value,
                          self.mom_value, self.dam_value, self.decay_value)

        # derive dtype from input
        assert param_value.dtype == self.dtype

        # TODO add branch operator
        # if self.decay_value != 0:
        if self.weight_decay.init_value != 0:
            singa.Axpy(self.decay_value.data, param_value.data, param_grad.data)

        if self.momentum.init_value != 0:
            if param_name not in self.moments:
                flag = param_value.device.graph_enabled()
                param_value.device.EnableGraph(False)
                self.moments[param_name] = tensor.zeros_like(param_value)
                param_value.device.EnableGraph(flag)

            buf = self.moments[param_name]
            buf *= self.mom_value
            alpha = 1.0 - self.dam_value
            singa.Axpy(alpha.data, param_grad.data, buf.data)

            if self.nesterov:
                singa.Axpy(self.mom_value.data, buf.data, param_grad.data)
            else:
                param_grad = buf

        minus_lr = 0.0 - self.lr_value
        singa.Axpy(minus_lr.data, param_grad.data, param_value.data)

    def step(self):
        # increment step counter, lr and moment
        super().step()
        mom_value = self.momentum(self.step_counter).as_type(self.dtype)
        dam_value = self.dampening(self.step_counter).as_type(self.dtype)
        decay_value = self.weight_decay(self.step_counter).as_type(self.dtype)
        self.mom_value.copy_from(mom_value)
        self.dam_value.copy_from(dam_value)
        self.decay_value.copy_from(decay_value)

    def get_states(self):
        states = super().get_states()
        if self.mom_value > 0:
            states[
                'moments'] = self.moments  # a dict for 1st order moments tensors
        return states

    def set_states(self, states):
        super().set_states(states)
        if 'moments' in states:
            self.moments = states['moments']
            self.mom_value = self.momentum(self.step_counter)


if __name__ == '__main__':
    # Use argparse to get command config: max_epoch, model, data, etc., for single gpu training
    parser = argparse.ArgumentParser(
        description='Training using the autograd and graph.')
    parser.add_argument(
        'model',
        choices=['cnn', 'resnet', 'xceptionnet', 'mlp', 'msmlp', 'alexnet'],
        default='cnn')
    parser.add_argument('data',
                        choices=['mnist', 'cifar10', 'cifar100'],
                        default='mnist')
    parser.add_argument('-p',
                        choices=['float32', 'float16'],
                        default='float32',
                        dest='precision')
    parser.add_argument('-m',
                        '--max-epoch',
                        default=3,
                        type=int,
                        help='maximum epochs',
                        dest='max_epoch')
    parser.add_argument('-b',
                        '--batch-size',
                        default=64,
                        type=int,
                        help='batch size',
                        dest='batch_size')
    parser.add_argument('-l',
                        '--learning-rate',
                        default=0.005,
                        type=float,
                        help='initial learning rate',
                        dest='lr')
    # Determine which gpu to use
    parser.add_argument('-i',
                        '--device-id',
                        default=0,
                        type=int,
                        help='which GPU to use',
                        dest='device_id')
    parser.add_argument('-g',
                        '--disable-graph',
                        default='True',
                        action='store_false',
                        help='disable graph',
                        dest='graph')
    parser.add_argument('-v',
                        '--log-verbosity',
                        default=0,
                        type=int,
                        help='logging verbosity',
                        dest='verbosity')

    args = parser.parse_args()

    mssgd = MSSGD(lr=args.lr, momentum=0.9, weight_decay=1e-5, dtype=singa_dtype[args.precision])
    run(0,
        1,
        args.device_id,
        args.max_epoch,
        args.batch_size,
        args.model,
        args.data,
        mssgd,
        args.graph,
        args.verbosity,
        precision=args.precision)