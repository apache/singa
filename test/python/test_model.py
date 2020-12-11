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

from singa import singa_wrap as singa_api
from singa.tensor import Tensor
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
        self.l1 = layer.Linear(a, b)
        self.l2 = layer.Linear(b, c)

    def forward(self, x):
        y = self.l1(x)
        y = self.l2(y)
        return y


class MyModel(model.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = layer.Conv2d(2, 2)
        self.bn1 = layer.BatchNorm2d(2)
        self.doublelinear1 = DoubleLinear(2, 4, 2)
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
        self.optimizer(loss)


class MLP(model.Model):

    def __init__(self, data_size=10, perceptron_size=100, num_classes=10):
        super(MLP, self).__init__()
        self.num_classes = num_classes
        self.dimension = 2

        self.relu = layer.ReLU()
        self.linear1 = layer.Linear(perceptron_size)
        self.linear2 = layer.Linear(num_classes)
        self.softmax_cross_entropy = layer.SoftMaxCrossEntropy()

    def forward(self, inputs):
        y = self.linear1(inputs)
        y = self.relu(y)
        y = self.linear2(y)
        return y

    def train_one_batch(self, x, y):
        out = self.forward(x)
        loss = self.softmax_cross_entropy(out, y)
        self.optimizer(loss)
        return out, loss

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer


# lstm testing
class LSTMModel3(model.Model):

    def __init__(self, hidden_size):
        super(LSTMModel3, self).__init__()
        self.lstm = layer.CudnnRNN(
            hidden_size=hidden_size,
            batch_first=True,
            #    return_sequences=True,
            use_mask=True)
        self.l1 = layer.Linear(2)
        self.optimizer = opt.SGD(0.1)

    def forward(self, x, seq_lengths):
        y = self.lstm(x, seq_lengths=seq_lengths)
        y = autograd.reshape(y, (y.shape[0], -1))
        y = self.l1(y)
        return y


class LSTMModel2(model.Model):

    def __init__(self, hidden_size, bidirectional, num_layers):
        super(LSTMModel2, self).__init__()
        self.lstm = layer.CudnnRNN(hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   bidirectional=bidirectional,
                                   return_sequences=False,
                                   rnn_mode='lstm',
                                   batch_first=True)
        self.optimizer = opt.SGD(0.1)

    def forward(self, x):
        return self.lstm(x)


class LSTMModel(model.Model):

    def __init__(self, hidden_size, seq_length, batch_size, bidirectional,
                 num_layers, return_sequences, rnn_mode, batch_first):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.return_sequences = return_sequences

        self.lstm = layer.CudnnRNN(hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   bidirectional=bidirectional,
                                   return_sequences=return_sequences,
                                   rnn_mode=rnn_mode,
                                   batch_first=batch_first)
        self.optimizer = opt.SGD(0.1)

    def forward(self, x):
        y = self.lstm(x)
        if self.return_sequences:
            y = autograd.reshape(y, (-1, self.seq_length * self.hidden_size))
        return y


class TestModelMethods(unittest.TestCase):

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_rnn_with_seq_lengths(self, dev=gpu_dev):
        bs = 2
        seq_length = 3
        hidden_size = 2
        em_size = 2
        x_np = np.array([[[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]],
                         [[0.3, 0.3], [0.4, 0.4], [0.0,
                                                   0.0]]]).astype(np.float32)
        y_np = np.array([[0.4, 0.4], [0.5, 0.5]]).astype(np.float32)
        seq_lengths_np = np.array([3, 2]).astype(np.int32)

        x = tensor.from_numpy(x_np)
        x.to_device(dev)
        y = tensor.from_numpy(y_np)
        y.to_device(dev)
        seq_lengths = tensor.from_numpy(seq_lengths_np)

        m = LSTMModel3(hidden_size)
        m.compile([x, seq_lengths],
                  is_train=True,
                  use_graph=False,
                  sequential=False)
        m.train()
        for i in range(10):
            out = m.forward(x, seq_lengths)
            loss = autograd.mse_loss(out, y)
            print("train l:", tensor.to_numpy(loss))
            m.optimizer(loss)
        m.eval()
        out = m.forward(x, seq_lengths)
        loss = autograd.mse_loss(out, y)
        print(" eval l:", tensor.to_numpy(loss))

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_lstm_model(self, dev=gpu_dev):
        hidden_size = 3
        seq_length = 2
        batch_size = 4
        feature_size = 3
        bidirectional = False
        directions = 2 if bidirectional else 1
        num_layers = 2
        out_size = hidden_size
        return_sequences = False
        batch_first = True
        rnn_mode = "lstm"

        # manual test case
        x_data = np.array([[[0, 0, 1], [0, 1, 0]], [[0, 1, 0], [1, 0, 0]],
                           [[0, 0, 1], [0, 1, 0]], [[1, 0, 0], [0, 0, 1]]],
                          dtype=np.float32).reshape(batch_size, seq_length,
                                                    hidden_size)  # bs, seq, fea
        if return_sequences:
            y_data = np.array(
                [[[0, 1, 0], [1, 0, 0]], [[1, 0, 0], [0, 0, 1]],
                 [[0, 1, 0], [1, 0, 0]], [[0, 0, 1], [0, 1, 0]]],
                dtype=np.float32).reshape(batch_size, seq_length,
                                          hidden_size)  # bs, hidden
            y_data.reshape(batch_size, -1)
        else:
            y_data = np.array([[1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]],
                              dtype=np.float32).reshape(
                                  batch_size, hidden_size)  # bs, hidden

        x = tensor.Tensor(device=dev, data=x_data)
        y_t = tensor.Tensor(device=dev, data=y_data)

        m = LSTMModel(hidden_size, seq_length, batch_size, bidirectional,
                      num_layers, return_sequences, rnn_mode, batch_first)
        m.compile([x], is_train=True, use_graph=False, sequential=False)

        m.train()
        for i in range(1000):
            y = m.forward(x)
            assert y.shape == y_t.shape
            loss = autograd.softmax_cross_entropy(y, y_t)
            if i % 100 == 0:
                print("loss", loss)
            m.optimizer(loss)

        m.eval()
        y = m.forward(x)
        loss = autograd.softmax_cross_entropy(y, y_t)
        print("eval loss", loss)


class TestModelSaveMethods(unittest.TestCase):

    def _save_states_load_states_helper(self, dev, graph_flag="False"):
        x_shape = (2, 2, 2, 2)
        x = tensor.PlaceHolder(x_shape, device=dev)

        m = MyModel()
        m.compile([x], is_train=True, use_graph=graph_flag, sequential=False)

        states = {
            "conv1.W":
                tensor.Tensor((2, 2, 2, 2), device=dev).set_value(0.1),
            "conv1.b":
                tensor.Tensor((2,), device=dev).set_value(0.2),
            "bn1.scale":
                tensor.Tensor((2,), device=dev).set_value(0.3),
            "bn1.bias":
                tensor.Tensor((2,), device=dev).set_value(0.4),
            "bn1.running_mean":
                tensor.Tensor((2,), device=dev).set_value(0.5),
            "bn1.running_var":
                tensor.Tensor((2,), device=dev).set_value(0.6),
            "doublelinear1.l1.W":
                tensor.Tensor((2, 4), device=dev).set_value(0.7),
            "doublelinear1.l1.b":
                tensor.Tensor((4,), device=dev).set_value(0.8),
            "doublelinear1.l2.W":
                tensor.Tensor((4, 2), device=dev).set_value(0.9),
            "doublelinear1.l2.b":
                tensor.Tensor((2,), device=dev).set_value(1.0)
        }

        m.set_states(states)
        states2 = m.get_states()
        for k in states2.keys():
            np.testing.assert_array_almost_equal(tensor.to_numpy(states[k]),
                                                 tensor.to_numpy(states2[k]))

        opt_state1 = tensor.Tensor((2, 10), device=dev).gaussian(1, 0.1)
        opt_state2 = tensor.Tensor((20, 2), device=dev).gaussian(0.1, 1)
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
        np.testing.assert_array_almost_equal(tensor.to_numpy(aux2["opt1"]),
                                             tensor.to_numpy(aux["opt1"]))
        np.testing.assert_array_almost_equal(tensor.to_numpy(aux2["opt2"]),
                                             tensor.to_numpy(aux["opt2"]))

        # snapshot states
        states3 = m.get_states()
        for k in states3.keys():
            np.testing.assert_array_almost_equal(tensor.to_numpy(states[k]),
                                                 tensor.to_numpy(states3[k]))

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_save_states_load_states_gpu(self):
        self._save_states_load_states_helper(gpu_dev, graph_flag=False)
        self._save_states_load_states_helper(gpu_dev, graph_flag=True)

    def test_save_states_load_states_cpu(self):
        self._save_states_load_states_helper(cpu_dev, graph_flag=False)
        self._save_states_load_states_helper(cpu_dev, graph_flag=True)


class TestPythonModule(unittest.TestCase):

    def to_categorical(self, y, num_classes):
        y = np.array(y, dtype="int")
        n = y.shape[0]
        categorical = np.zeros((n, num_classes))
        categorical[np.arange(n), y] = 1
        return categorical

    def generate_data(self, dev, num=400):
        f = lambda x: (5 * x + 1)

        x = np.random.uniform(-1, 1, num)
        y = f(x) + 2 * np.random.randn(len(x))

        self.label = np.asarray([5 * a + 1 > b for (a, b) in zip(x, y)])
        self.data = np.array([[a, b] for (a, b) in zip(x, y)], dtype=np.float32)
        self.label = self.to_categorical(self.label, 2).astype(np.float32)

        self.inputs = Tensor(data=self.data, device=dev)
        self.target = Tensor(data=self.label, device=dev)

    def get_params(self, model):
        params = model.get_params()
        self.w0 = params['linear1.W']
        self.b0 = params['linear1.b']
        self.w1 = params['linear2.W']
        self.b1 = params['linear2.b']

        self.W0 = tensor.to_numpy(self.w0)
        self.B0 = tensor.to_numpy(self.b0)
        self.W1 = tensor.to_numpy(self.w1)
        self.B1 = tensor.to_numpy(self.b1)

    def numpy_forward(self, inputs):
        self.x1 = np.matmul(inputs, self.W0)
        self.x2 = np.add(self.x1, self.B0)
        self.x3 = np.maximum(self.x2, 0)
        self.x4 = np.matmul(self.x3, self.W1)
        self.x5 = np.add(self.x4, self.B1)
        return self.x5

    def numpy_train_one_batch(self, inputs, y):
        # forward propagation
        out = self.numpy_forward(inputs)

        # softmax cross entropy loss
        exp_out = np.exp(out - np.max(out, axis=-1, keepdims=True))
        self.softmax = exp_out / np.sum(exp_out, axis=-1, keepdims=True)
        loss = np.sum(y * np.log(self.softmax)) / -self.softmax.shape[0]

        # optimize
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
        return out, loss

    def setUp(self):
        self.sgd = opt.SGD(lr=0.05)

        cpu_dev.ResetGraph()
        if singa_api.USE_CUDA:
            gpu_dev.ResetGraph()

    def tearDown(self):
        cpu_dev.ResetGraph()
        if singa_api.USE_CUDA:
            gpu_dev.ResetGraph()

    def _forward_helper(self, dev, is_train, use_graph, sequential):
        self.generate_data(dev)
        model = MLP(self.sgd)
        model.compile([self.inputs],
                      is_train=is_train,
                      use_graph=use_graph,
                      sequential=sequential)

        self.get_params(model)

        out = model(self.inputs)
        np_out = self.numpy_forward(self.data)

        np.testing.assert_array_almost_equal(tensor.to_numpy(out), np_out)

    def _train_one_batch_helper(self, dev, is_train, use_graph, sequential):
        self.generate_data(dev)
        model = MLP(num_classes=2)
        model.set_optimizer(self.sgd)
        model.compile([self.inputs],
                      is_train=is_train,
                      use_graph=use_graph,
                      sequential=sequential)

        self.get_params(model)

        out, loss = model(self.inputs, self.target)
        np_out, np_loss = self.numpy_train_one_batch(self.data, self.label)

        np.testing.assert_array_almost_equal(tensor.to_numpy(out), np_out)
        np.testing.assert_array_almost_equal(tensor.to_numpy(loss), np_loss)
        np.testing.assert_array_almost_equal(tensor.to_numpy(self.w0), self.W0)
        np.testing.assert_array_almost_equal(tensor.to_numpy(self.b0), self.B0)
        np.testing.assert_array_almost_equal(tensor.to_numpy(self.w1), self.W1)
        np.testing.assert_array_almost_equal(tensor.to_numpy(self.b1), self.B1)

    def test_forward_cpu(self):
        self._forward_helper(cpu_dev, False, True, False)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_forward_gpu(self):
        self._forward_helper(gpu_dev, False, True, False)

    def test_evaluate_cpu(self):
        self._forward_helper(cpu_dev, False, False, False)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_evaluate_gpu(self):
        self._forward_helper(gpu_dev, False, False, False)

    def test_train_one_batch_cpu(self):
        self._train_one_batch_helper(cpu_dev, True, True, False)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_train_one_batch_gpu(self):
        self._train_one_batch_helper(gpu_dev, True, True, False)

    def test_without_graph_cpu(self):
        self._train_one_batch_helper(cpu_dev, True, False, False)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_without_graph_gpu(self):
        self._train_one_batch_helper(gpu_dev, True, False, False)

    def test_run_in_serial_cpu(self):
        self._train_one_batch_helper(cpu_dev, True, True, True)

    @unittest.skipIf(not singa_api.USE_CUDA, 'CUDA is not enabled')
    def test_run_in_serial_gpu(self):
        self._train_one_batch_helper(gpu_dev, True, True, True)


if __name__ == '__main__':
    unittest.main()
