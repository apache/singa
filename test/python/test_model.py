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

import os
import math
import unittest
import numpy as np
import time

from singa import singa_wrap as singa_api
from singa import autograd
from singa import tensor
from singa import device
from singa import layer
from singa import model
from singa import opt

from cuda_helper import gpu_dev, cpu_dev

class DoubleLinear(layer.Layer):
    def __init__(self, a, b, c):
        super(DoubleLinear, self).__init__()
        self.l1 = layer.Linear(a,b)
        self.l2 = layer.Linear(b,c)

    def forward(self, x):
        y = self.l1(x)
        y = self.l2(y)
        return y

class MyModel(model.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = layer.Conv2d(2,2)
        self.bn1 = layer.BatchNorm2d(2)
        self.doublelinear1 = DoubleLinear(2,4,2)
        self.optimizer = opt.SGD()

    def forward(self, x):
        y = self.conv1(x)
        y = self.bn1(y)
        y = autograd.reshape(y, (y.shape[0], -1))
        y = self.doublelinear1(y)
        return y

    def train_one_batch(self, x, y):
        y_ = self.forward(x)
        l = self.loss(y_, y)
        self.optim(l)
        return y_, l

    def loss(self, out, ty):
        return autograd.softmax_cross_entropy(out, ty)

    def optim(self, loss):
        self.optimizer.backward_and_update(loss)

class TestTensorMethods(unittest.TestCase):
    def _save_states_load_states_helper(self, dev, graph_flag="False"):
        x_shape =(2, 2, 2, 2)
        x = tensor.PlaceHolder(x_shape, device=dev)

        m = MyModel()
        m.on_device(dev)

        m.compile([x], is_train=True, use_graph=graph_flag, sequential=False)

        states = {
                "MyModel.conv1.W": tensor.Tensor((2, 2, 2, 2), device=dev).set_value(0.1),
                "MyModel.conv1.b": tensor.Tensor((2,), device=dev).set_value(0.2),
                "MyModel.bn1.scale": tensor.Tensor((2,), device=dev).set_value(0.3),
                "MyModel.bn1.bias": tensor.Tensor((2,), device=dev).set_value(0.4),
                "MyModel.bn1.running_mean": tensor.Tensor((2,), device=dev).set_value(0.5),
                "MyModel.bn1.running_var": tensor.Tensor((2,), device=dev).set_value(0.6),
                "MyModel.doublelinear1.l1.W": tensor.Tensor((2, 4), device=dev).set_value(0.7),
                "MyModel.doublelinear1.l1.b": tensor.Tensor((4,), device=dev).set_value(0.8),
                "MyModel.doublelinear1.l2.W": tensor.Tensor((4, 2), device=dev).set_value(0.9),
                "MyModel.doublelinear1.l2.b": tensor.Tensor((2,), device=dev).set_value(1.0)}

        m.set_states(states)
        states2 = m.get_states()
        for k in states2.keys():
            np.testing.assert_array_almost_equal(tensor.to_numpy(states[k]), tensor.to_numpy(states2[k]))


        opt_state1 = tensor.Tensor((2,10), device=dev).gaussian(1, 0.1)
        opt_state2 = tensor.Tensor((20,2), device=dev).gaussian(0.1, 1)
        aux = {"opt1": opt_state1, "opt2": opt_state2}

        # save snapshot1
        zip_fp = 'snapshot1_%s.zip' % self._testMethodName
        if os.path.exists(zip_fp):
            os.remove(zip_fp)
        m.save_states(zip_fp, aux)

        # do some training, states changes
        cx = tensor.Tensor(x_shape, device=dev).gaussian(1, 1)
        cy = tensor.Tensor((2, 2), device=dev).gaussian(1, 1)
        mini_batch_size = 10
        for i in range(mini_batch_size):
            m.train_one_batch(cx, cy)

        # restore snapshot
        aux2 = m.load_states(zip_fp)
        np.testing.assert_array_almost_equal(tensor.to_numpy(aux2["opt1"]), tensor.to_numpy(aux["opt1"]))
        np.testing.assert_array_almost_equal(tensor.to_numpy(aux2["opt2"]), tensor.to_numpy(aux["opt2"]))

        # snapshot states
        states3 = m.get_states()
        for k in states3.keys():
            np.testing.assert_array_almost_equal(tensor.to_numpy(states[k]), tensor.to_numpy(states3[k]))


    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_save_states_load_states_gpu(self):
        self._save_states_load_states_helper(gpu_dev, graph_flag=False)
        self._save_states_load_states_helper(gpu_dev, graph_flag=True)

    def test_save_states_load_states_cpu(self):
        self._save_states_load_states_helper(cpu_dev, graph_flag=False)
        self._save_states_load_states_helper(cpu_dev, graph_flag=True)

if __name__ == '__main__':
    unittest.main()
