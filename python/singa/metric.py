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
'''This module includes a set of metric classes for evaluating the model's
performance. The specific metric classes could be converted from C++
implmentation or implemented directly using Python.


Example usage::

    from singa import tensor
    from singa import metric

    x = tensor.Tensor((3, 5))
    x.uniform(0, 1)  # randomly genearte the prediction activation
    x = tensor.SoftMax(x)  # normalize the prediction into probabilities
    y = tensor.from_numpy(np.array([0, 1, 3], dtype=np.int))  # set the truth

    f = metric.Accuracy()
    acc = f.evaluate(x, y)  # averaged accuracy over all 3 samples in x

'''


from . import singa_wrap as singa
import tensor
import numpy as np


class Metric(object):
    '''Base metric class.

    Subclasses that wrap the C++ loss classes can use the inherited foward,
    and evaluate functions of this base class. Other subclasses need
    to override these functions. Users need to feed in the **predictions** and
    ground truth to get the metric values.
    '''

    def __init__(self):
        self.swig_metric = None

    def forward(self, x, y):
        '''Compute the metric for each sample.

        Args:
            x (Tensor): predictions, one row per sample
            y (Tensor): ground truth values, one row per sample

        Returns:
            a tensor of floats, one per sample
        '''
        return tensor.from_raw_tensor(
            self.swig_metric.Forward(x.singa_tensor, y.singa_tensor))

    def evaluate(self, x, y):
        '''Compute the averaged metric over all samples.

        Args:
            x (Tensor): predictions, one row per sample
            y (Tensor): ground truth values, one row per sample
        Returns:
            a float value for the averaged metric
        '''
        return self.swig_metric.Evaluate(x.singa_tensor, y.singa_tensor)


class Accuracy(Metric):
    '''Compute the top one accuracy for single label prediction tasks.

    It calls the C++ functions to do the calculation.
    '''
    def __init__(self):
        self.swig_metric = singa.Accuracy()


class Precision(Metric):
    '''Make the top-k labels of max probability as the prediction

    Compute the precision against the groundtruth labels
    '''
    def __init__(self, top_k):
        self.top_k = top_k



    def forward(self, x, y):
        '''Compute the precision for each sample.

        Convert tensor to numpy for computation

        Args:
            x (Tensor): predictions, one row per sample
            y (Tensor): ground truth labels, one row per sample

        Returns:
            a tensor of floats, one per sample
        '''

        dev = x.device
        x.to_host()
        y.to_host()

        x_np = tensor.to_numpy(x)
        y_np = tensor.to_numpy(y)

        pred_np = np.argsort(-x_np)[:,0:self.top_k] #Sort in descending order

        tmp_np = np.zeros(pred_np.shape, dtype=np.float32)

        for i in range(pred_np.shape[0]):
            tmp_np[i] = y_np[i,pred_np[i]]

        prcs_np = np.average(tmp_np, axis=1)

        prcs = tensor.from_numpy(prcs_np)

        x.to_device(dev)
        y.to_device(dev)
        prcs.to_device(dev)

        return prcs


    def evaluate(self, x, y):
        '''Compute the averaged precision over all samples.

        Args:
            x (Tensor): predictions, one row per sample
            y (Tensor): ground truth values, one row per sample
        Returns:
            a float value for the averaged metric
        '''

        return tensor.average(self.forward(x,y))


class Precision(Metric):
    '''Make the top-k labels of max probability as the prediction

    Compute the precision against the groundtruth labels
    '''
    def __init__(self, top_k):
        self.top_k = top_k



    def forward(self, x, y):
        '''Compute the precision for each sample.

        Convert tensor to numpy for computation

        Args:
            x (Tensor): predictions, one row per sample
            y (Tensor): ground truth labels, one row per sample

        Returns:
            a tensor of floats, one per sample
        '''

        dev = x.device
        x.to_host()
        y.to_host()

        x_np = tensor.to_numpy(x)
        y_np = tensor.to_numpy(y)

        pred_np = np.argsort(-x_np)[:,0:self.top_k] #Sort in descending order

        tmp_np = np.zeros(pred_np.shape, dtype=np.float32)

        for i in range(pred_np.shape[0]):
            tmp_np[i] = y_np[i,pred_np[i]]

        prcs_np = np.average(tmp_np, axis=1)

        prcs = tensor.from_numpy(prcs_np)

        x.to_device(dev)
        y.to_device(dev)
        prcs.to_device(dev)

        return prcs


    def evaluate(self, x, y):
        '''Compute the averaged precision over all samples.

        Args:
            x (Tensor): predictions, one row per sample
            y (Tensor): ground truth values, one row per sample
        Returns:
            a float value for the averaged metric
        '''

        return tensor.average(self.forward(x,y))
