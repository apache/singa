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

# lstm testing
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
    def test_lstm_model_varying_bs_seq_length(self, dev=gpu_dev):
        hidden_size = 6
        bidirectional = False
        num_layers = 1
        m = LSTMModel2(hidden_size, bidirectional, num_layers)
        x = tensor.Tensor(shape=(2, 3, 4), device=dev)
        x.gaussian(0, 1)
        y = tensor.Tensor(shape=(2, hidden_size), device=dev)
        y.gaussian(0, 1)
        m.compile([x], is_train=True, use_graph=False, sequential=False)
        m.train()
        for i in range(1000):
            out = m.forward(x)
            loss = autograd.mse_loss(out, y)
            if i % 50 == 0:
                print("l:", loss)
            m.optimizer.backward_and_update(loss)

        # bs changed
        bs = 1
        seq = 2
        x2 = tensor.Tensor(shape=(bs, seq, 4), device=dev)
        x2.gaussian(0, 1)
        y2 = tensor.Tensor(shape=(bs, hidden_size), device=dev)
        y2.gaussian(0, 1)

        out = m.forward(x2)
        loss = autograd.mse_loss(out, y2)
        print("test l:", loss)
        m.optimizer.backward_and_update(loss)

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
            m.optimizer.backward_and_update(loss)

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
            "MyModel.conv1.W":
                tensor.Tensor((2, 2, 2, 2), device=dev).set_value(0.1),
            "MyModel.conv1.b":
                tensor.Tensor((2,), device=dev).set_value(0.2),
            "MyModel.bn1.scale":
                tensor.Tensor((2,), device=dev).set_value(0.3),
            "MyModel.bn1.bias":
                tensor.Tensor((2,), device=dev).set_value(0.4),
            "MyModel.bn1.running_mean":
                tensor.Tensor((2,), device=dev).set_value(0.5),
            "MyModel.bn1.running_var":
                tensor.Tensor((2,), device=dev).set_value(0.6),
            "MyModel.doublelinear1.l1.W":
                tensor.Tensor((2, 4), device=dev).set_value(0.7),
            "MyModel.doublelinear1.l1.b":
                tensor.Tensor((4,), device=dev).set_value(0.8),
            "MyModel.doublelinear1.l2.W":
                tensor.Tensor((4, 2), device=dev).set_value(0.9),
            "MyModel.doublelinear1.l2.b":
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


if __name__ == '__main__':
    unittest.main()
