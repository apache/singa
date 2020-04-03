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
        self.w1.gaussian(0.0, 0.1)
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
        self.x1 = np.matmul(inputs, self.W0)
        self.x2 = np.add(self.x1, self.B0)
        self.x3 = np.maximum(self.x2, 0)
        self.x4 = np.matmul(self.x3, self.W1)
        self.x5 = np.add(self.x4, self.B1)
        return self.x5

    def numpy_loss(self, out, y):
        exp_out = np.exp(out - np.max(out, axis=-1, keepdims=True))
        self.softmax = exp_out / np.sum(exp_out, axis=-1, keepdims=True)

        loss = np.sum(y * np.log(self.softmax)) / -self.softmax.shape[0]

        return loss

    def numpy_optim(self, loss):
        # calculate gradients
        label_sum = np.sum(self.label, axis=-1)
        dloss = self.softmax - self.label / label_sum.reshape(
            label_sum.shape[0], 1)
        dloss /= self.softmax.shape[0]

        dx5 = dloss
        db1 = np.sum(dloss, 0)

        dx4 = np.matmul(dx5, self.W1.T)
        dw1 = np.matmul(self.x3.T, dx5)

        dx3 = dx4 * (self.x3 > 0)

        dx2 = dx3
        db0 = np.sum(dx3, 0)

        dx1 = np.matmul(dx2, self.W0.T)
        dw0 = np.matmul(self.data.T, dx2)

        # update all the params
        self.W0 -= 0.05 * dw0
        self.B0 -= 0.05 * db0
        self.W1 -= 0.05 * dw1
        self.B1 -= 0.05 * db1

    def setUp(self):
        self.sgd = opt.SGD(lr=0.05)

        self.generate_data(400)

        cpu_dev.ResetGraph()
        gpu_dev.ResetGraph()

    def test_forward(self):
        for dev in [cpu_dev, gpu_dev]:
            model = MLP(self.sgd)
            model.train()
            model.on_device(dev)
            self.get_numpy_params(model)

            out = model(self.inputs)

            np_out = self.numpy_forward(self.data)

            np.testing.assert_array_almost_equal(tensor.to_numpy(out), np_out)

    def test_forward_loss(self):
        for dev in [cpu_dev, gpu_dev]:
            model = MLP(self.sgd)
            model.train()
            model.on_device(dev)
            self.get_numpy_params(model)

            out = model(self.inputs)
            loss = model.loss(out, self.target)

            np_out = self.numpy_forward(self.data)
            np_loss = self.numpy_loss(np_out, self.label)

            np.testing.assert_array_almost_equal(tensor.to_numpy(out), np_out)
            np.testing.assert_array_almost_equal(tensor.to_numpy(loss), np_loss)

    def test_forward_loss_optim(self):
        for dev in [cpu_dev, gpu_dev]:
            model = MLP(self.sgd)
            model.train()
            model.on_device(dev)
            self.get_numpy_params(model)

            out = model(self.inputs)
            loss = model.loss(out, self.target)
            model.optim(loss)

            np_out = self.numpy_forward(self.data)
            np_loss = self.numpy_loss(np_out, self.label)
            self.numpy_optim(np_loss)

            np.testing.assert_array_almost_equal(tensor.to_numpy(out), np_out)
            np.testing.assert_array_almost_equal(tensor.to_numpy(loss), np_loss)
            np.testing.assert_array_almost_equal(tensor.to_numpy(model.w0),
                                                 self.W0)
            np.testing.assert_array_almost_equal(tensor.to_numpy(model.b0),
                                                 self.B0)
            np.testing.assert_array_almost_equal(tensor.to_numpy(model.w1),
                                                 self.W1)
            np.testing.assert_array_almost_equal(tensor.to_numpy(model.b1),
                                                 self.B1)

    def test_train_without_graph(self):
        for dev in [cpu_dev, gpu_dev]:
            model = MLP(self.sgd)
            model.train()
            model.on_device(dev)
            model.graph(False)
            self.get_numpy_params(model)

            out = model(self.inputs)
            loss = model.loss(out, self.target)
            model.optim(loss)

            np_out = self.numpy_forward(self.data)
            np_loss = self.numpy_loss(np_out, self.label)
            self.numpy_optim(np_loss)

            np.testing.assert_array_almost_equal(tensor.to_numpy(out), np_out)
            np.testing.assert_array_almost_equal(tensor.to_numpy(loss), np_loss)
            np.testing.assert_array_almost_equal(tensor.to_numpy(model.w0),
                                                 self.W0)
            np.testing.assert_array_almost_equal(tensor.to_numpy(model.b0),
                                                 self.B0)
            np.testing.assert_array_almost_equal(tensor.to_numpy(model.w1),
                                                 self.W1)
            np.testing.assert_array_almost_equal(tensor.to_numpy(model.b1),
                                                 self.B1)

    def test_run_in_serial(self):
        for dev in [cpu_dev, gpu_dev]:
            model = MLP(self.sgd)
            model.train()
            model.on_device(dev)
            model.graph(True, False)
            self.get_numpy_params(model)

            out = model(self.inputs)
            loss = model.loss(out, self.target)
            model.optim(loss)

            np_out = self.numpy_forward(self.data)
            np_loss = self.numpy_loss(np_out, self.label)
            self.numpy_optim(np_loss)

            np.testing.assert_array_almost_equal(tensor.to_numpy(out), np_out)
            np.testing.assert_array_almost_equal(tensor.to_numpy(loss), np_loss)
            np.testing.assert_array_almost_equal(tensor.to_numpy(model.w0),
                                                 self.W0)
            np.testing.assert_array_almost_equal(tensor.to_numpy(model.b0),
                                                 self.B0)
            np.testing.assert_array_almost_equal(tensor.to_numpy(model.w1),
                                                 self.W1)
            np.testing.assert_array_almost_equal(tensor.to_numpy(model.b1),
                                                 self.B1)

    def test_evaluate(self):
        for dev in [cpu_dev, gpu_dev]:
            model = MLP(self.sgd)
            model.eval()
            model.on_device(dev)
            self.get_numpy_params(model)

            out = model(self.inputs)

            np_out = self.numpy_forward(self.data)

            np.testing.assert_array_almost_equal(tensor.to_numpy(out), np_out)


if __name__ == '__main__':
    unittest.main()
