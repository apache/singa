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
'''
Loss module includes a set of training loss implmentations. Some are converted
from C++ implementation, and the rest are implemented directly using python
Tensor.

Example usage::

    from singa import tensor
    from singa import loss
    import numpy as np

    x = tensor.Tensor((3, 5))
    x.uniform(0, 1)  # randomly generate the prediction activation
    y = tensor.from_numpy(np.array([0, 1, 3], dtype=np.int))  # set the truth

    f = loss.SoftmaxCrossEntropy()
    l = f.forward(True, x, y)  # l is tensor with 3 loss values
    g = f.backward()  # g is a tensor containing all gradients of x w.r.t l
'''
from __future__ import division
from __future__ import absolute_import
from builtins import object

from . import singa_wrap as singa
from . import tensor
from .proto import model_pb2


class Loss(object):
    '''Base loss class.

    Subclasses that wrap the C++ loss classes can use the inherited foward,
    backward, and evaluate functions of this base class. Other subclasses need
    to override these functions
    '''

    def __init__(self):
        self.swig_loss = None

    def forward(self, flag, x, y):
        '''Compute the loss values.

        Args:
            flag: kTrain/kEval or bool. If it is kTrain/True, then the backward
                function must be called before calling forward again.
            x (Tensor): the prediction Tensor
            y (Tensor): the ground truch Tensor, x.shape[0] must = y.shape[0]

        Returns:
            a tensor of floats for the loss values, one per sample
        '''
        if type(flag) is bool:
            if flag:
                flag = model_pb2.kTrain
            else:
                flag = model_pb2.kEval
        return tensor.from_raw_tensor(
            self.swig_loss.Forward(flag, x.data, y.data))

    def backward(self):
        '''
        Returns:
            the grad of x w.r.t. the loss
        '''
        return tensor.from_raw_tensor(self.swig_loss.Backward())

    def evaluate(self, flag, x, y):  # TODO(wangwei) remove flag
        '''
        Args:
            flag (int): must be kEval, to be removed
            x (Tensor): the prediction Tensor
            y (Tensor): the ground truth Tnesor

        Returns:
            the averaged loss for all samples in x.
        '''
        if type(flag) is bool:
            if flag:
                flag = model_pb2.kTrain
            else:
                flag = model_pb2.kEval

        return self.swig_loss.Evaluate(flag, x.data, y.data)


class SoftmaxCrossEntropy(Loss):
    '''This loss function is a combination of SoftMax and Cross-Entropy loss.

    It converts the inputs via SoftMax function and then
    computes the cross-entropy loss against the ground truth values.

    For each sample, the ground truth could be a integer as the label index;
    or a binary array, indicating the label distribution. The ground truth
    tensor thus could be a 1d or 2d tensor.
    The data/feature tensor could 1d (for a single sample) or 2d for a batch of
    samples.
    '''

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
        self.swig_loss = singa.SoftmaxCrossEntropy()


class SigmoidCrossEntropy(Loss):
    '''This loss evaluates the cross-entropy loss between the prediction and the
    truth values with the prediction probability generated from Sigmoid.
    '''

    def __init__(self, epsilon=1e-8):
        super(SigmoidCrossEntropy, self).__init__()
        self.truth = None
        self.prob = None
        self.epsilon = epsilon  # to avoid log(x) with x being too small

    def forward(self, flag, x, y):
        '''loss is -yi * log pi - (1-yi) log (1-pi), where pi=sigmoid(xi)

        Args:
            flag (bool): true for training; false for evaluation
            x (Tensor): the prediction Tensor
            y (Tensor): the truth Tensor, a binary array value per sample

        Returns:
            a Tensor with one error value per sample
        '''
        p = tensor.sigmoid(x)
        if flag:
            self.truth = y
            self.prob = p
        np = 1 - p
        p += (p < self.epsilon) * self.epsilon
        np += (np < self.epsilon) * self.epsilon
        l = (y - 1) * tensor.log(np) - y * tensor.log(p)
        # TODO(wangwei): add unary operation -Tensor
        return tensor.average(l, axis=1)

    def backward(self):
        ''' Compute the gradient of loss w.r.t to x.

        Returns:
            dx = pi - yi.
        '''
        assert self.truth is not None, 'must call forward in a prior'
        dx = self.prob - self.truth
        self.truth = None
        return dx

    def evaluate(self, flag, x, y):
        '''Compuate the averaged error.

        Returns:
            a float value as the averaged error
        '''
        l = self.forward(False, x, y)
        return l.l1()


class SquaredError(Loss):
    '''This loss evaluates the squared error between the prediction and the
    truth values.

    It is implemented using Python Tensor operations.
    '''

    def __init__(self):
        super(SquaredError, self).__init__()
        self.err = None

    def forward(self, flag, x, y):
        '''Compute the error as 0.5 * ||x-y||^2.

        Args:
            flag (int): kTrain or kEval; if kTrain, then the backward must be
                called before calling forward again.
            x (Tensor): the prediction Tensor
            y (Tensor): the truth Tensor, an integer value per sample, whose
                value is [0, x.shape[1])

        Returns:
            a Tensor with one error value per sample
        '''
        self.err = x - y
        return tensor.square(self.err) * 0.5

    def backward(self):
        '''Compute the gradient of x w.r.t the error.

        Returns:
            x - y
        '''
        return self.err

    def evaluate(self, flag, x, y):
        '''Compuate the averaged error.

        Returns:
            a float value as the averaged error
        '''
        return tensor.sum(tensor.square(x - y)) * 0.5 / x.size()
