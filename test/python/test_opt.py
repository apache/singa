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
from __future__ import division

import math
import unittest
import numpy as np
import functools

from singa import tensor
from singa import singa_wrap as singa
from singa import opt

from cuda_helper import gpu_dev, cpu_dev


def assertTensorEqual(x, y, decimal=6):
    assert x.shape == y.shape
    assert x.dtype == y.dtype
    assert x.device.id() == y.device.id()
    d = x.device
    x.to_host()
    y.to_host()
    np.testing.assert_array_almost_equal(x.data.GetFloatValue(int(x.size())),
                                         y.data.GetFloatValue(int(y.size())),
                                         decimal)
    x.to_device(d)
    y.to_device(d)


def on_cpu_gpu(func):

    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        func(*args, dev=cpu_dev, **kwargs)
        if singa.USE_CUDA:
            func(*args, dev=gpu_dev, **kwargs)

    return wrapper_decorator


class TestDecayScheduler(unittest.TestCase):

    def test_exponential_decay_cpu(self):
        lr = opt.ExponentialDecay(0.1, 2, 0.5, True)
        sgd1 = opt.SGD(lr=lr)
        for i in range(5):
            np.testing.assert_array_almost_equal(tensor.to_numpy(sgd1.lr_value),
                                                 [0.1 * 0.5**(i // 2)])
            sgd1.step()

    def test_exponential_decay_no_staircase_cpu(self):
        lr = opt.ExponentialDecay(0.1, 2, 0.5, False)
        sgd1 = opt.SGD(lr=lr)
        for i in range(5):
            np.testing.assert_array_almost_equal(tensor.to_numpy(sgd1.lr_value),
                                                 [0.1 * 0.5**(i / 2)])
            sgd1.step()

    @on_cpu_gpu
    def test_const_decay_scheduler(self, dev):
        c1 = opt.Constant(0.2)
        step = tensor.Tensor((1,), device=dev).set_value(0)
        lr_val = c1(step)
        np.testing.assert_array_almost_equal(tensor.to_numpy(c1(step)), [0.2])
        step += 1
        np.testing.assert_array_almost_equal(tensor.to_numpy(c1(step)), [0.2])


class TestOptimizer(unittest.TestCase):

    @on_cpu_gpu
    def test_optimizer(self, dev):
        o1 = opt.Optimizer(0.1)

        # test step
        o1.step()
        o1.step()

        # test get states
        s1 = o1.get_states()
        self.assertAlmostEqual(s1['step_counter'], 2)

        # test set states
        s2 = {'step_counter': 5}
        o1.set_states(s2)
        np.testing.assert_array_almost_equal(tensor.to_numpy(o1.step_counter),
                                             [5])

    @on_cpu_gpu
    def test_sgd_const_lr(self, dev=cpu_dev):
        cpu_dev.EnableGraph(False)
        sgd1 = opt.SGD(lr=0.1)
        w_shape = (2, 3)
        w = tensor.Tensor(w_shape, device=dev).set_value(0.1)
        g = tensor.Tensor(w_shape, device=dev).set_value(0.1)

        w_step1 = w - 0.1 * g
        sgd1.apply(w.name, w, g)

        assertTensorEqual(w, w_step1)

    @on_cpu_gpu
    def test_RMSProp_const_lr(self, dev=cpu_dev):
        cpu_dev.EnableGraph(False)
        opt1 = opt.RMSProp(lr=0.1)
        w_shape = (2, 3)
        w = tensor.Tensor(w_shape, device=dev).set_value(0.1)
        g = tensor.Tensor(w_shape, device=dev).set_value(0.1)

        # running_average = running_average * rho + param_grad * param_grad * (1 - rho)
        # param_value = param_value - lr * param_grad / sqrt(running_average + epsilon)

        running_average = 0.1 * tensor.square(g)
        tmp = running_average + 1e-8
        tmp = tensor.sqrt(tmp)
        tmp = g / tmp

        w_step1 = w - 0.1 * tmp
        opt1.apply(w.name, w, g)

        assertTensorEqual(w, w_step1)

    @on_cpu_gpu
    def test_AdaGrad_const_lr(self, dev=cpu_dev):
        cpu_dev.EnableGraph(False)
        opt1 = opt.AdaGrad(lr=0.1)
        w_shape = (2, 3)
        w = tensor.Tensor(w_shape, device=dev).set_value(0.1)
        g = tensor.Tensor(w_shape, device=dev).set_value(0.1)

        # history = history + param_grad * param_grad
        # param_value = param_value - lr * param_grad / sqrt(history + epsilon)

        history = tensor.square(g)
        tmp = history + 1e-8
        tmp = tensor.sqrt(tmp)
        tmp = g / tmp

        w_step1 = w - 0.1 * tmp
        opt1.apply(w.name, w, g)

        assertTensorEqual(w, w_step1)

    @on_cpu_gpu
    def test_Adam_const_lr(self, dev=cpu_dev):
        cpu_dev.EnableGraph(False)
        opt1 = opt.Adam(lr=0.1)
        w_shape = (2, 3)
        w = tensor.Tensor(w_shape, device=dev).set_value(1.0)
        g = tensor.Tensor(w_shape, device=dev).set_value(0.1)

        # m := beta_1 * m + (1 - beta_1) * grad
        # v := beta_2 * v + (1 - beta_2) * grad * grad
        # m_norm = m / (1 - beta_1 ^ step)
        # v_norm = v / (1 - beta_2 ^ step)
        # param := param - (lr * m_norm) / ( sqrt(v_norm) + epsilon) )

        m = 0.1 * g
        tmp = tensor.square(g)
        v = 0.001 * tmp

        m_norm = m / 0.1
        v_norm = v / 0.001

        tmp = tensor.sqrt(v_norm) + 1e-8
        tmp = m_norm / tmp

        w_step1 = w - 0.1 * tmp
        opt1.apply(w.name, w, g)

        assertTensorEqual(w, w_step1, decimal=5)

    @on_cpu_gpu
    def test_sgd_const_lr_momentum(self, dev=cpu_dev):
        sgd1 = opt.SGD(lr=0.1, momentum=0.9)
        w_shape = (2, 3)
        w = tensor.Tensor(w_shape, device=dev).set_value(0.1)
        g = tensor.Tensor(w_shape, device=dev).set_value(0.01)

        w_step1 = w - 0.1 * g
        buf = g

        sgd1.apply(w.name, w, g)
        sgd1.step()

        assertTensorEqual(w, w_step1)

        buf = g + buf * 0.9
        w_step2 = w - 0.1 * buf

        sgd1.apply(w.name, w, g)

        assertTensorEqual(w, w_step2)

    @on_cpu_gpu
    def test_sgd_const_lr_momentum_weight_decay(self, dev=cpu_dev):
        sgd1 = opt.SGD(lr=0.1, weight_decay=0.2)
        w_shape = (2, 3)
        w = tensor.Tensor(w_shape, device=dev).set_value(0.1)
        g = tensor.Tensor(w_shape, device=dev).set_value(0.01)

        w_step1 = w - 0.1 * (g + 0.2 * w)

        sgd1.apply(w.name, w, g)

        assertTensorEqual(w, w_step1)

    # @on_cpu_gpu
    def test_sgd_const_lr_momentum_nesterov(self, dev=cpu_dev):
        sgd1 = opt.SGD(lr=0.1, momentum=0.9, nesterov=True)
        w_shape = (2, 3)
        w = tensor.Tensor(w_shape, device=dev).set_value(0.1)
        g = tensor.Tensor(w_shape, device=dev).set_value(0.1)

        buf = g
        w_step1 = w - 0.1 * (g + 0.9 * buf)

        sgd1.apply(w.name, w, g)

        assertTensorEqual(w, w_step1)


if __name__ == '__main__':
    unittest.main()
