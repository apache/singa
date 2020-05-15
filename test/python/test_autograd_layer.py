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
import logging
import sys

from singa import tensor
from singa import singa_wrap as singa_api
from singa import autograd
from singa.proto import core_pb2

from cuda_helper import gpu_dev, cpu_dev


class TestLinearLayer(unittest.TestCase):
    def test_param_init(self):
        x = tensor.Tensor((10,2)).gaussian(1, 2)

        li = autograd.Linear(2, 3)

        li.set_params_initializer(
            b=lambda x: x.gaussian(0,1),
            W=lambda x: x.set_value(0.0)
        )
        y=li(x)
        params = li.get_params()
        pass

    def test_init_in_and_out(self):
        li1 = autograd.Linear(2, 3)
        li2 = autograd.Linear(3)

        for li in [li1,li2]:
            x = tensor.Tensor((10,2)).gaussian(1, 2)
            W = tensor.Tensor((2,3)).gaussian(1, 2)
            b = tensor.Tensor((3,)).gaussian(1, 2)
            li.set_params(W=W,b=b)
            (W,b) = li.get_params()
            y=li(x)
        pass

    def test_init_by_set_params(self):
        x = tensor.Tensor((10,2)).gaussian(1, 2)
        W = tensor.Tensor((2,3)).gaussian(1, 2)
        b = tensor.Tensor((3,)).gaussian(1, 2)

        li1 = autograd.Linear(3)

        # logging.debug(li1.__dict__)
        li1.set_params(W=W,b=b)
        (W,b) = li1.get_params()
        y=li1(x)
        pass

    def test_init_by_forward(self):
        x = tensor.Tensor((10,2)).gaussian(1, 2)
        W = tensor.Tensor((2,3)).gaussian(1, 2)
        b = tensor.Tensor((3,)).gaussian(1, 2)

        li1 = autograd.Linear(3)

        y=li1(x)
        (W,b) = li1.get_params()
        pass

class TestRNNLayer(unittest.TestCase):
    def test_set_param(self):
        batch_size = 2
        feature_size = 3
        sequence_size = 4
        hidden_size = 5

        rnn1 = autograd.RNN(hidden_size)
        rnn2 = autograd.RNN(feature_size, hidden_size)

        for rnn in [rnn1, rnn2]:
            xs = []
            for i in range(sequence_size):
                x = tensor.Tensor((batch_size,feature_size)).gaussian(1, 2)
                xs.append(x)
                pass

            h0 = tensor.Tensor((1, hidden_size)).gaussian(1, 2)
            Wx = tensor.Tensor((feature_size, hidden_size)).gaussian(1, 2)
            Wh = tensor.Tensor((hidden_size, hidden_size)).gaussian(1, 2)
            b = tensor.Tensor((hidden_size, )).gaussian(1, 2)

            rnn.set_params(Wx=Wx, Wh=Wh, b=b)
            (Wx, Wh, b) = rnn.get_params()

            (ys, h) = rnn(xs, h0)
            # for y in ys:
                # print("y shape", y.shape)
            # logging.debug("h shape", h.shape)
        pass
    def test_set_init(self):
        batch_size = 2
        feature_size = 3
        sequence_size = 4
        hidden_size = 5

        rnn = autograd.RNN(feature_size, hidden_size)
        rnn.set_params_initializer(
                Wx=lambda x: x.set_value(0.0),
                Wh=lambda x: x.set_value(0.0),
                b=lambda x: x.gaussian(0,1)
        )

        xs = []
        for i in range(sequence_size):
            x = tensor.Tensor((batch_size,feature_size)).gaussian(1, 2)
            xs.append(x)
            pass

        h0 = tensor.Tensor((1, hidden_size)).gaussian(1, 2)

        (ys, h) = rnn(xs, h0)

class TestLSTMLayer(unittest.TestCase):
    def get_xs(self, sequence_size, batch_size, feature_size):
        xs = []
        for i in range(sequence_size):
            x = tensor.Tensor((batch_size,feature_size)).gaussian(1, 2)
            xs.append(x)
        return xs

    def test_forward(self):
        batch_size = 2
        feature_size = 3
        sequence_size = 4
        hidden_size = 5
        lstm1 = autograd.LSTM(hidden_size)
        h0 = tensor.Tensor((1, hidden_size)).gaussian(1, 2)
        c0 = tensor.Tensor((1, hidden_size)).gaussian(1, 2)
        y, h, c = lstm1(self.get_xs(sequence_size, batch_size, feature_size), (h0,c0))

    def test_set_init(self):
        batch_size = 2
        feature_size = 3
        sequence_size = 4
        hidden_size = 5
        lstm = autograd.LSTM(hidden_size)
        lstm.set_params_initializer(
                Wx=lambda x: x.set_value(0.0),
                Wh=lambda x: x.set_value(0.0),
                Bx=lambda x: x.gaussian(0,1),
                Bh=lambda x: x.gaussian(0,1)
        )
        h0 = tensor.Tensor((1, hidden_size)).gaussian(1, 2)
        c0 = tensor.Tensor((1, hidden_size)).gaussian(1, 2)
        y, h, c = lstm(self.get_xs(sequence_size, batch_size, feature_size), (h0,c0))
        pass

logging.basicConfig(stream=sys.stderr, level=logging.INFO)
if __name__ == '__main__':
    unittest.main()
