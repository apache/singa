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

def assertTensorEqual(x,y):
    assert x.shape == y.shape
    assert x.dtype == y.dtype
    assert x.device.id() == y.device.id()
    d = x.device
    x.to_host()
    y.to_host()
    np.testing.assert_array_almost_equal(
        x.data.GetFloatValue(int(x.size())),
        y.data.GetFloatValue(int(y.size())))
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
            np.testing.assert_array_almost_equal(tensor.to_numpy(sgd1.lr_value), [0.1*0.5**(i//2)])
            sgd1.step()

    @on_cpu_gpu
    def test_const_decay_scheduler(self, dev):
        c1 = opt.Constant(0.2)
        step = tensor.Tensor((1,), device=dev).set_value(0)
        lr_val = c1(step)
        np.testing.assert_array_almost_equal(tensor.to_numpy(c1(step)) , [0.2])
        step+=1
        np.testing.assert_array_almost_equal(tensor.to_numpy(c1(step)) , [0.2])

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
        np.testing.assert_array_almost_equal( tensor.to_numpy(o1.step_counter), [5])

    @on_cpu_gpu
    def test_sgd_const_lr(self, dev):
        sgd1 = opt.SGD(lr=0.1)
        w_shape=(2,3)
        w = tensor.Tensor(w_shape, device=dev).set_value(0.1)
        g = tensor.Tensor(w_shape, device=dev).set_value(0.01)

        sgd1.apply(w, w, g)
        sgd1.step()

        assertTensorEqual(w, w-0.1*g)

    @on_cpu_gpu
    def test_sgd_const_lr_momentum(self, dev=cpu_dev):
        sgd1 = opt.SGD(lr=0.1,momentum=0.9)
        w_shape=(2,3)
        w = tensor.Tensor(w_shape, device=dev).set_value(0.1)
        g = tensor.Tensor(w_shape, device=dev).set_value(0.01)

        sgd1.apply(w, w, g)
        sgd1.step()

        w_step1 = w-0.1*g
        buf = g
        assertTensorEqual(w,w_step1)

        sgd1.apply(w, w, g)
        sgd1.step()

        buf = g + buf*0.9
        w_step2 = w-0.1*buf
        assertTensorEqual(w, w_step2)

    @on_cpu_gpu
    def test_sgd_const_lr_momentum_weight_decay(self, dev):
        sgd1 = opt.SGD(lr=0.1, weight_decay=0.2)
        w_shape=(2,3)
        w = tensor.Tensor(w_shape, device=dev).set_value(0.1)
        g = tensor.Tensor(w_shape, device=dev).set_value(0.01)

        sgd1.apply(w, w, g)
        sgd1.step()

        w_step1 = w-0.1*(g+0.2*w)
        assertTensorEqual(w,w_step1)

    @on_cpu_gpu
    def test_sgd_const_lr_momentum_nesterov(self, dev):
        sgd1 = opt.SGD(lr=0.1, momentum=0.9, nesterov=True)
        w_shape=(2,3)
        w = tensor.Tensor(w_shape, device=dev).set_value(0.1)
        g = tensor.Tensor(w_shape, device=dev).set_value(0.01)

        sgd1.apply(w, w, g)
        sgd1.step()

        w_step1 = w-0.1*(g*0.9)
        assertTensorEqual(w,w_step1)

if __name__ == '__main__':
    unittest.main()
