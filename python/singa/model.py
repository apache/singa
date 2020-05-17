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
This script includes Model class for python users
to use Computational Graph in their model.
'''

from functools import wraps

from singa import tensor
from singa import autograd
from . import singa_wrap as singa
from .device import get_default_device

import gc


class Graph(type):

    def buffer_operation(func):

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.graph_mode and self.training:
                if not self.buffered:
                    # buffer operations
                    self._device.EnableGraph(True)
                    self._results = func(self, *args, **kwargs)
                    self._device.Sync()
                    self._device.EnableGraph(False)
                    self.buffered = True

                    # deconstruct Operations before running the entire graph
                    if self._results:
                        if isinstance(self._results, list):
                            for _matrix in self._results:
                                if isinstance(_matrix, tensor.Tensor):
                                    _matrix.creator = None
                        elif isinstance(self._results, tensor.Tensor):
                            self._results.creator = None

                    # make sure all Operations are deallocated
                    gc.collect()

                # run graph
                self._device.RunGraph(self.sequential)
                return self._results
            else:
                return func(self, *args, **kwargs)

        return wrapper

    def __new__(cls, name, bases, attr):
        attr["train_one_batch"] = Graph.buffer_operation(attr["train_one_batch"])

        return super(Graph, cls).__new__(cls, name, bases, attr)


class Model(object, metaclass=Graph):
    """ Base class for your neural network models.

    Example usage::

        import numpy as np
        from singa import opt
        from singa import tensor
        from singa import device
        from singa import autograd
        from singa import layer
        from singa import model

        class MyModel(model.Model):
            def __init__(self):
                super(MyModel, self).__init__()

                self.conv1 = layer.Conv2d(1, 20, 5, padding=0)
                self.conv2 = layer.Conv2d(20, 50, 5, padding=0)

                self.sgd = opt.SGD(lr=0.01)

            def forward(self, x):
                y = self.conv1(x)
                y = self.conv2(y)
                return y

            def train_one_batch(self, x, y):
                out = self.forward(x)
                loss = autograd.softmax_cross_entropy(out, y)
                self.sgd.backward_and_update(loss)
                return out, loss

    """

    def __init__(self):
        """
        Initializes internal Model state
        """
        self.training = True
        self.buffered = False
        self.graph_mode = True
        self.sequential = False
        self.initialized = False
        self._device = get_default_device()

        self._results = None

    def compile(self, inputs, is_train=True, use_graph=False, sequential=False):
        self._device.EnableGraph(True)
        self.forward(*inputs)
        self._device.EnableGraph(False)
        self._device.ResetGraph()
        autograd.training = is_train
        self.training = is_train
        self.graph_mode = use_graph
        self.sequential = sequential

    def forward(self, *input):
        """Defines the computation performed at every call.

        Should be overridden by all subclasses.

        Args:
            *input: the input training data for the model

        Returns:
            out: the outputs of the forward propagation.
        """
        raise NotImplementedError

    def train_one_batch(self, *input):
        raise NotImplementedError

    def train(self, mode=True):
        """Set the model in evaluation mode.

        Args:
            mode(bool): when mode is True, this model will enter training mode
        """
        self.training = mode
        autograd.training = mode

    def eval(self):
        """Sets the model in evaluation mode.
        """
        self.train(mode=False)
        autograd.training = False

    def graph(self, mode=True, sequential=False):
        """ Turn on the computational graph. Specify execution mode.

        Args:
            mode(bool): when mode is True, model will use computational graph
            sequential(bool): when sequential is True, model will execute ops
            in the graph follow the order of joining the graph
        """
        self.graph_mode = mode
        self.sequential = sequential

    def on_device(self, device):
        """Sets the target device.

        The following training will be performed on that device.

        Args:
            device(Device): the target device
        """
        self._device = device

    def __get_name__(self):
        return self.__class__.__name__

    def __call__(self, *input, **kwargs):
        if self.training:
            return self.train_one_batch(*input, **kwargs)
        else:
            return self.forward(*input, **kwargs)
