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
This script includes Module class for python users
to use Computational Graph in their model.
'''

from functools import wraps

from singa import autograd
from . import singa_wrap as singa
from .device import get_default_device

import gc


class Graph(type):

    def buffer_operation(func):

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.graph_mode and self.training:
                name = func.__name__
                if name not in self._called:
                    # tag this function
                    self._called.add(name)
                    # buffer operations
                    self._device.EnableGraph(True)
                    ret = func(self, *args, **kwargs)
                    self._device.Sync()
                    self._device.EnableGraph(False)
                    # deconstruct Operations before running the entire graph
                    if name == 'optim':
                        for fname in self._results:
                            if isinstance(self._results[fname], list):
                                for _matrix in self._results[fname]:
                                    _matrix.creator = None
                            else:
                                self._results[fname].creator = None
                        # make sure all Operations are deallocated
                        gc.collect()
                    # add result tensor
                    self._results[name] = ret
                    # run graph
                    self._device.RunGraph(self.sequential)
                    self.initialized = True
                    return ret

                return self._results[name]
            else:
                return func(self, *args, **kwargs)

        return wrapper

    def __new__(cls, name, bases, attr):
        attr["forward"] = Graph.buffer_operation(attr["forward"])
        attr["loss"] = Graph.buffer_operation(attr["loss"])
        attr["optim"] = Graph.buffer_operation(attr["optim"])

        return super(Graph, cls).__new__(cls, name, bases, attr)


class Module(object, metaclass=Graph):
    """ Base class for your neural network modules.

    Example usage::

        import numpy as np
        from singa import opt
        from singa import tensor
        from singa import device
        from singa import autograd
        from singa.module import Module

        class Model(Module):
            def __init__(self):
                super(Model, self).__init__()

                self.conv1 = autograd.Conv2d(1, 20, 5, padding=0)
                self.conv2 = autograd.Conv2d(20, 50, 5, padding=0)

                self.sgd = opt.SGD(lr=0.01)

            def forward(self, x):
                y = self.conv1(x)
                y = self.conv2(y)
                return y

            def loss(self, out, y):
                return autograd.softmax_cross_entropy(out, y)

            def optim(self, loss):
                self.sgd.backward_and_update(loss)

    """

    def __init__(self):
        """
        Initializes internal Module state
        """
        self.training = True
        self.graph_mode = True
        self.sequential = False
        self.initialized = False
        self._device = get_default_device()

        self._results = {}
        self._called = set()

    def forward(self, *input):
        """Defines the computation performed at every call.

        Should be overridden by all subclasses.

        Args:
            *input: the input training data for the module

        Returns:
            out: the outputs of the forward propagation.
        """
        raise NotImplementedError

    def loss(self, *args, **kwargs):
        """Defines the loss function performed when training the module.
        """
        pass

    def optim(self, *args, **kwargs):
        """Defines the optim function for backward pass.
        """
        pass

    def train(self, mode=True):
        """Set the module in evaluation mode.

        Args:
            mode(bool): when mode is True, this module will enter training mode
        """
        self.training = mode
        autograd.training = True

    def eval(self):
        """Sets the module in evaluation mode.
        """
        self.train(mode=False)
        autograd.training = False

    def graph(self, mode=True, sequential=False):
        """ Turn on the computational graph. Specify execution mode.

        Args:
            mode(bool): when mode is True, module will use computational graph
            sequential(bool): when sequential is True, module will execute ops
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
        if self.graph_mode and self.training:
            if self.initialized == True:
                self._device.RunGraph(self.sequential)

        return self.forward(*input, **kwargs)
