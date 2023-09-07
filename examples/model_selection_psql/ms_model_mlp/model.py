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

from singa import layer
from singa import model
from singa import tensor
from singa import opt
from singa import device
from singa.autograd import Operator
from singa.layer import Layer
from singa import singa_wrap as singa
import argparse
import numpy as np

np_dtype = {"float16": np.float16, "float32": np.float32}

singa_dtype = {"float16": tensor.float16, "float32": tensor.float32}

#### self-defined loss begin

### from autograd.py
class SumError(Operator):

    def __init__(self):
        super(SumError, self).__init__()
        # self.t = t.data

    def forward(self, x):
        # self.err = singa.__sub__(x, self.t)
        self.data_x = x
        # sqr = singa.Square(self.err)
        # loss = singa.SumAll(sqr)
        loss = singa.SumAll(x)
        # self.n = 1
        # for s in x.shape():
        #     self.n *= s
        # loss /= self.n
        return loss

    def backward(self, dy=1.0):
        # dx = self.err
        dev = device.get_default_device()
        dx = tensor.Tensor(self.data_x.shape, dev, singa_dtype['float32'])
        dx.copy_from_numpy(np.ones(self.data_x.shape))
        # dx *= float(2 / self.n)
        dx *= dy
        return dx

def se_loss(x):
    # assert x.shape == t.shape, "input and target shape different: %s, %s" % (
    #     x.shape, t.shape)
    return SumError()(x)[0]

### from layer.py
class SumErrorLayer(Layer):
    """
    Generate a MeanSquareError operator
    """

    def __init__(self):
        super(SumErrorLayer, self).__init__()

    def forward(self, x):
        return se_loss(x)

#### self-defined loss end

class MSMLP(model.Model):

    def __init__(self, data_size=10, perceptron_size=100, num_classes=10, layer_hidden_list=[10,10,10,10]):
        super(MSMLP, self).__init__()
        self.num_classes = num_classes
        self.dimension = 2

        self.relu = layer.ReLU()
        self.linear1 = layer.Linear(layer_hidden_list[0])
        self.linear2 = layer.Linear(layer_hidden_list[1])
        self.linear3 = layer.Linear(layer_hidden_list[2])
        self.linear4 = layer.Linear(layer_hidden_list[3])
        self.linear5 = layer.Linear(num_classes)
        self.softmax_cross_entropy = layer.SoftMaxCrossEntropy()
        self.sum_error = SumErrorLayer()
