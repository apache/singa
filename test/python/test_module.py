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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

import unittest
import numpy as np
from builtins import str

from singa import opt
from singa import device
from singa import tensor
from singa import module
from singa import autograd
from singa.tensor import Tensor

from cuda_helper import cpu_dev, gpu_dev


class MLP(module.Module):

    def __init__(self, optimizer):
        super(MLP, self).__init__()

        self.w0 = Tensor(shape=(2, 3), requires_grad=True, stores_grad=True)
        self.b0 = Tensor(shape=(3,), requires_grad=True, stores_grad=True)
        self.w1 = Tensor(shape=(3, 2), requires_grad=True, stores_grad=True)
        self.b1 = Tensor(shape=(2,), requires_grad=True, stores_grad=True)

        self.w0.gaussian(0.0, 0.1)
        self.b0.set_value(0.0)
        self.w1.set_value(0.0, 0.1)
        self.b1.set_value(0.0)

        self.optimizer = optimizer

    def forward(self, inputs):
        x = autograd.matmul(inputs, self.w0)
        x = autograd.add_bias(x, self.b0)
        x = autograd.relu(x)
        x = autograd.matmul(x, self.w1)
        x = autograd.add_bias(x, self.b1)
        return x

    def loss(self, out, target):
        return autograd.softmax_cross_entropy(out, target)

    def optim(self, loss):
        return self.optimizer.backward_and_update(loss)


class TestPythonModule(unittest.TestCase):

    def to_categorical(self, y, num_classes):
        y = np.array(y, dtype="int")
        n = y.shape[0]
        categorical = np.zeros((n, num_classes))
        categorical[np.arange(n), y] = 1
        return categorical

    def generate_data(self, num=400):
        f = lambda x: (5 * x + 1)

        x = np.random.uniform(-1, 1, num)
        y = f(x) + 2 * np.random.randn(len(x))

        self.label = np.asarray([5 * a + 1 > b for (a, b) in zip(x, y)])
        self.data = np.array([[a, b] for (a, b) in zip(x, y)], dtype=np.float32)
        self.label = self.to_categorical(self.label, 2).astype(np.float32)

        self.inputs = Tensor(data=self.data)
        self.target = Tensor(data=self.label)

    def get_numpy_params(self, model):
        self.W0 = tensor.to_numpy(model.w0)
        self.B0 = tensor.to_numpy(model.b0)
        self.W1 = tensor.to_numpy(model.w1)
        self.B1 = tensor.to_numpy(model.b1)

    def numpy_forward(self, inputs):
        res = np.matmul(inputs, self.W0)
        res = np.add(res, self.B0)
        res = np.maximum(res, 0)
        res = np.matmul(res, self.W1)
        res = np.add(res, self.B1)
        return res

    def numpy_loss(self, out, y):
        pass

    def numpy_optim(self, loss):
        pass

    def setUp(self):
        self.sgd = opt.SGD(lr=0.05)

        self.generate_data(400)

        cpu_dev.ResetGraph()
        gpu_dev.ResetGraph()

    def test_forward(self):
        for dev in [cpu_dev, gpu_dev]:
            model = MLP(self.sgd)
            model.on_device(dev)
            self.get_numpy_params(model)

            x = model(self.inputs)

            np_x = self.numpy_forward(self.data)

            np.testing.assert_array_almost_equal(tensor.to_numpy(x), np_x)

    def test_forward_loss(self):
        pass

    def test_forward_loss_optim(self):
        pass

    def test_without_graph(self):
        pass

    def test_run_in_serial(self):
        pass

    def test_train(self):
        pass

    def test_validate(self):
        pass


if __name__ == '__main__':
    unittest.main()
