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
    from singa.proto import model_pb2

    x = tensor.Tensor((3, 5))
    x.uniform(0, 1)  # randomly genearte the prediction activation
    y = tensor.from_numpy(np.array([0, 1, 3], dtype=np.int))  # set the truth

    f = loss.SoftmaxCrossEntropy()
    l = f.forward(model_pb2.kTrain, x, y)  # l is tensor with 3 loss values
    g = f.backward()  # g is a tensor containing all gradients of x w.r.t l
'''


from . import singa_wrap as singa
import tensor


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
            flag (int): kTrain or kEval. If it is kTrain, then the backward
                function must be called before calling forward again.
            x (Tensor): the prediction Tensor
            y (Tensor): the ground truch Tensor, x.shape[0] must = y.shape[0]

        Returns:
            a tensor of floats for the loss values, one per sample
        '''
        return tensor.from_raw_tensor(
            self.swig_loss.Forward(flag, x.singa_tensor, y.singa_tensor))

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
        return self.swig_loss.Evaluate(flag, x.singa_tensor, y.singa_tensor)


class SoftmaxCrossEntropy(Loss):
    '''This loss function is a combination of SoftMax and Cross-Entropy loss.

    It converts the inputs via SoftMax function and then
    computes the cross-entropy loss against the ground truth values.
    '''

    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
        self.swig_loss = singa.SoftmaxCrossEntropy()


class SquaredError(Loss):
    '''This loss evaluates the squared error between the prediction and the
    truth values.

    It is implemented using Python Tensor operations.
    '''
    def __init__(self):
        super(SquareLoss, self).__init__()
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
        return tensor.sum(tensor.square(x - y) * 0.5) / x.size()
