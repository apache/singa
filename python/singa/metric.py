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
    '''Compute the top one accuracy for singel label prediction tasks.

    It calls the C++ functions to do the calculation.
    '''
    def __init__(self):
        self.swig_metric = singa.Accuracy()
