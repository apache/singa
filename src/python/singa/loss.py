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
""" Python wrappers for optimizers implemented by C++."""

from . import singa_wrap as singa
import tensor


class Loss(object):

    def __init__(self):
        self.swig_loss = None

    def forward(self, flag, x, y):
        """Return a tensor of floats, one per sample"""
        return tensor.from_raw_tensor(
            self.swig_loss.Forward(flag, x.singa_tensor, y.singa_tensor))

    def backward(self):
        """Return the grad of x w.r.t. the loss obj"""
        return tensor.from_raw_tensor(self.swig_loss.Backward())

    def evaluate(self, flag, x, y):
        """Return the averaged loss for all samples in x"""
        return self.swig_loss.Evaluate(flag, x.singa_tensor, y.singa_tensor)


class SoftmaxCrossEntropy(Loss):

    def __init__(self):
        self.swig_loss = singa.SoftmaxCrossEntropy()
