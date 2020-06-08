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
import time
import numpy as np
import json
import zipfile
import os

from singa import tensor
from singa import autograd
from singa import layer
from .tensor import Tensor
from . import singa_wrap as singa

import gc


class ModelMeta(layer.LayerMeta):

    def buffer_operation(func):

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if self.graph_mode and self.training:
                if len(args) == 0:
                    raise ValueError('expect at least one input tensor')

                if isinstance(args[0], list):
                    assert isinstance(
                        args[0][0],
                        Tensor), ('function expects PlaceHolders or Tensors')
                    dev = args[0][0].device
                else:
                    assert isinstance(
                        args[0],
                        Tensor), ('function expects PlaceHolders or Tensors')
                    dev = args[0].device

                if not self._buffered:
                    # buffer operations
                    dev.EnableGraph(True)
                    self._results = func(self, *args, **kwargs)
                    dev.Sync()
                    dev.EnableGraph(False)
                    self._buffered = True

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
                dev.RunGraph(self.sequential)
                return self._results
            else:
                return func(self, *args, **kwargs)

        return wrapper

    def __new__(cls, name, bases, attr):
        if 'train_one_batch' in attr:
            attr['train_one_batch'] = ModelMeta.buffer_operation(
                attr['train_one_batch'])

        return super(ModelMeta, cls).__new__(cls, name, bases, attr)


class Model(layer.Layer, metaclass=ModelMeta):
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

                self.softmax_cross_entropy = layer.SoftMaxCrossEntropy()
                self.conv1 = layer.Conv2d(1, 20, 5, padding=0)
                self.conv2 = layer.Conv2d(20, 50, 5, padding=0)
                self.sgd = opt.SGD(lr=0.01)

            def forward(self, x):
                y = self.conv1(x)
                y = self.conv2(y)
                return y

            def train_one_batch(self, x, y):
                out = self.forward(x)
                loss = self.softmax_cross_entropy(out, y)
                self.sgd(loss)
                return out, loss

    """

    # save load states constant
    TENSOR_DICT_FILENAME = '/tensor_dict.npz'
    STATES_ATTR_FILENAME = '/states_attr.json'
    MODEL_STATE_TYPE = 0
    AUX_STATE_TYPE = 1

    def __init__(self):
        """
        Initializes internal Model state
        """
        super(Model, self).__init__()

        self.training = True
        self.graph_mode = True
        self.sequential = False
        self._buffered = False
        self._results = None

    def compile(self, inputs, is_train=True, use_graph=False, sequential=False):
        assert len(inputs) > 0 and isinstance(inputs[0], Tensor), (
            'compile function expects PlaceHolders or Tensors')

        dev = inputs[0].device
        dev.EnableGraph(True)
        self.forward(*inputs)
        dev.EnableGraph(False)
        dev.ResetGraph()

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

    def graph(self, mode=True, sequential=False):
        """ Turn on the computational graph. Specify execution mode.

        Args:
            mode(bool): when mode is True, model will use computational graph
            sequential(bool): when sequential is True, model will execute ops
            in the graph follow the order of joining the graph
        """
        self.graph_mode = mode
        self.sequential = sequential

    def __get_name__(self):
        return self.__class__.__name__

    def __call__(self, *input, **kwargs):
        if self.training:
            return self.train_one_batch(*input, **kwargs)
        else:
            return self.forward(*input, **kwargs)

    def save_states(self, fpath, aux_states={}):
        """Save states.

        Args:
            fpath: output file path (without the extension)
            aux_states(dict): values are standard data types or Tensor,
                              e.g., epoch ID, learning rate, optimizer states
        """
        assert not os.path.isfile(fpath), (
            "Failed to save states, %s is already existed." % fpath)

        states = self.get_states()

        # save states data and attr
        tensor_dict = {}
        states_attr = {}
        for k, v in states.items():
            assert isinstance(v, tensor.Tensor), "Only tensor state is allowed"
            tensor_dict[k] = tensor.to_numpy(v)
            states_attr[k] = {
                'state_type': self.MODEL_STATE_TYPE,
                'shape': v.shape,
                'dtype': v.dtype
            }

        for k, v in aux_states.items():
            assert isinstance(v,
                              tensor.Tensor), "Only tensor aux state is allowed"
            tensor_dict[k] = tensor.to_numpy(v)
            states_attr[k] = {
                'state_type': self.AUX_STATE_TYPE,
                'shape': v.shape,
                'dtype': v.dtype
            }

        # save to files
        timestamp = time.time()
        tmp_dir = '/tmp/singa_save_states_%s' % timestamp
        os.mkdir(tmp_dir)
        tensor_dict_fp = tmp_dir + self.TENSOR_DICT_FILENAME
        states_attr_fp = tmp_dir + self.STATES_ATTR_FILENAME

        np.savez(tensor_dict_fp, **tensor_dict)

        with open(states_attr_fp, 'w') as fp:
            json.dump(states_attr, fp)

        compression = zipfile.ZIP_DEFLATED
        with zipfile.ZipFile(fpath, mode="w") as zf:
            zf.write(tensor_dict_fp,
                     os.path.basename(tensor_dict_fp),
                     compress_type=compression)
            zf.write(states_attr_fp,
                     os.path.basename(states_attr_fp),
                     compress_type=compression)

        # clean up tmp files
        os.remove(tensor_dict_fp)
        os.remove(states_attr_fp)
        os.rmdir(tmp_dir)

    def load_states(self, fpath):
        """Load the model states and auxiliary states from disk.

        Usage:
            m = MyModel()
            m.compile(...)
            aux_states = m.load_states('mymodel.zip')

        Args:
            path: input file path (without the extension)
        Returns:
            dict
        """

        assert os.path.isfile(fpath), (
            "Failed to load states, %s is not exist." % fpath)

        timestamp = time.time()
        tmp_dir = '/tmp/singa_load_states_%s' % timestamp
        os.mkdir(tmp_dir)

        with zipfile.ZipFile(fpath, 'r') as zf:
            zf.extractall(tmp_dir)

        tensor_dict_fp = tmp_dir + self.TENSOR_DICT_FILENAME
        states_attr_fp = tmp_dir + self.STATES_ATTR_FILENAME

        with open(states_attr_fp) as f:
            states_attr = json.load(f)

        tensor_dict = np.load(tensor_dict_fp)

        # restore singa tensor from numpy
        model_states = dict()
        aux_states = dict()

        for k in tensor_dict.files:
            if states_attr[k]['state_type'] == self.MODEL_STATE_TYPE:
                model_states[k] = tensor.from_numpy(tensor_dict[k])
            elif states_attr[k]['state_type'] == self.AUX_STATE_TYPE:
                aux_states[k] = tensor.from_numpy(tensor_dict[k])

        # restore model_states
        self.set_states(model_states)

        # clean up tmp files
        os.remove(tensor_dict_fp)
        os.remove(states_attr_fp)
        os.rmdir(tmp_dir)
        return aux_states
