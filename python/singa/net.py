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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""
Nerual net class for constructing the nets using layers and providing access
functions for net info, e.g., parameters.


Example usages::

    from singa import net as ffnet
    from singa import metric
    from singa import loss
    from singa import layer
    from singa import device

    # create net and add layers
    net = ffnet.FeedForwardNet(loss.SoftmaxCrossEntropy(), metric.Accuracy())
    net.add(layer.Conv2D('conv1', 32, 5, 1, input_sample_shape=(3,32,32,)))
    net.add(layer.Activation('relu1'))
    net.add(layer.MaxPooling2D('pool1', 3, 2))
    net.add(layer.Flatten('flat'))
    net.add(layer.Dense('dense', 10))

    # init parameters
    for p in net.param_values():
        if len(p.shape) == 0:
            p.set_value(0)
        else:
            p.gaussian(0, 0.01)

    # move net onto gpu
    dev = device.create_cuda_gpu()
    net.to_device(dev)

    # training (skipped)

    # do prediction after training
    x = tensor.Tensor((2, 3, 32, 32), dev)
    x.uniform(-1, 1)
    y = net.predict(x)
    print tensor.to_numpy(y)
"""
from __future__ import print_function
from __future__ import absolute_import

from builtins import zip
from builtins import str
from builtins import object
import numpy as np
import os

from .proto.model_pb2 import kTrain, kEval
from .__init__ import __version__
from . import tensor
from . import layer
from . import snapshot

try:
    import pickle
except ImportError:
    import cPickle as pickle


'''For display training information, e.g L1 value of layer data'''
verbose = False


class FeedForwardNet(object):

    def __init__(self, loss=None, metric=None):
        '''Representing a feed-forward neural net.

        Args:
            loss, a Loss instance. Necessary training
            metric, a Metric instance. Necessary for evaluation
        '''
        self.loss = loss
        self.metric = metric
        self.layers = []
        self.src_of_layer = {}
        self.dst_of_layer = None
        self.ordered_layers = None
        self.out_sample_shape_of_layer = {}

    def to_device(self, dev):
        '''Move the net onto the given device, including
        all parameters and intermediate data.
        '''
        for lyr in self.layers:
            lyr.to_device(dev)

    def add(self, lyr, src=None):
        """Append a layer into the layer list.

        This function will get the sample shape from the src layers to setup the
        newly added layer. For the first layer, it is setup outside. The calling
        function should ensure the correctness of the layer order. If src is
        None, the last layer is the src layer. If there are multiple src layers,
        the src is a list of the src layers.

        Args:
            lyr (Layer): the layer to be added
            src (Layer): the source layer of lyr
        """
        if src is not None:
            if isinstance(src, layer.Layer):
                assert src.has_setup is True, 'the source layer must be set up'
                self.src_of_layer[lyr.name] = [src]
            else:
                assert type(src) == list, 'the src must be a list of layers'
                self.src_of_layer[lyr.name] = src
                # print 'merge------', len(src)
        else:
            assert len(self.layers) > 0 or lyr.has_setup, \
                'Source layers are needed to set up this layer'
            if len(self.layers) > 0:
                self.src_of_layer[lyr.name] = [self.layers[-1]]
            else:
                self.src_of_layer[lyr.name] = []
        if lyr.has_setup is False:
            in_shape = []
            for src in self.src_of_layer[lyr.name]:
                shapes = self.out_sample_shape_of_layer[src.name]
                assert len(shapes) > 0, \
                    'Cannot get output shape of layer %s' % lyr.name
                in_shape.append(shapes[0])
                shapes.pop(0)
            if len(in_shape) == 1:
                lyr.setup(in_shape[0])
            else:
                lyr.setup(in_shape)
        out_shape = lyr.get_output_sample_shape()
        if type(out_shape[0]) is tuple:
            self.out_sample_shape_of_layer[lyr.name] = out_shape
        else:
            self.out_sample_shape_of_layer[lyr.name] = [out_shape]
        self.layers.append(lyr)
        print((lyr.name, out_shape))
        return lyr

    def param_values(self):
        '''Return a list of tensors for all parameters'''
        values = []
        layers = self.layers
        if self.ordered_layers is not None:
            layers = self.ordered_layers
        for lyr in layers:
            values.extend(lyr.param_values())
        return values

    def param_specs(self):
        '''Return a list of ParamSpec for all parameters'''
        specs = []
        layers = self.layers
        if self.ordered_layers is not None:
            layers = self.ordered_layers
        for lyr in layers:
            specs.extend(lyr.param_specs)
        return specs

    def param_names(self):
        '''Return a list for the names of all params'''
        return [spec.name for spec in self.param_specs()]

    def train(self, x, y):
        '''Run BP for one iteration.
        This method is deprecated. It is only kept for backward compatibility.
        The name of this method is confusing since it does not update parameters.
        Please use backprob() instead.
        The back progagation algorithm computes gradients but it does not train.
        '''
        return self.backprob(x, y)

    def backprob(self, x, y):
        '''Run BP for one iteration.

        Currently only support nets with a single output layer, and a single
        loss objective and metric.
        For multiple outputs (with multiple loss/metric), please manually
        call forward, compute loss/metric and call backward. backward() is also
        more memory efficient than this function.

        Args:
            x: input data, a single input Tensor or a dict: layer name -> Tensor
            y: label data, a single input Tensor.
        Returns:
            gradients of parameters and the loss and metric values.
        '''
        out = self.forward(kTrain, x)
        l = self.loss.forward(kTrain, out, y)
        g = self.loss.backward()
        g /= x.shape[0]
        m = None
        if self.metric is not None:
            m = self.metric.evaluate(out, y)
        grads = []  # store all gradient tensors; memory inefficient
        for _, _, grad, _ in self.backward(g):
            grads.extend(grad[::-1])
        return grads[::-1], (l.l1(), m)

    def evaluate(self, x, y):
        '''Evaluate the loss and metric of the given data.

        Currently only support nets with a single output layer, and a single
        loss objective and metric.
        TODO(wangwei) consider multiple loss objectives and metrics.

        Args:
            x: input data, a single input Tensor or a dict: layer name -> Tensor
            y: label data, a single input Tensor.
        '''
        out = self.forward(kEval, x)
        l = None
        m = None
        assert self.loss is not None or self.metric is not None,\
            'Cannot do evaluation, as neither loss nor metic is set'
        if self.loss is not None:
            l = self.loss.evaluate(kEval, out, y)
        if self.metric is not None:
            m = self.metric.evaluate(out, y)
        return l, m

    def predict(self, x):
        '''Forward the input data through each layer to get the values of the
        output layers.

        Currently only support nets with a single output layer
        TODO(yujian) to handle multiple outputs from the network

        Args:
            x: input data, a single input Tensor or a dict: layer name -> Tensor

        Returns:
            a single output tensor as the prediction result.

        '''

        xx = self.forward(kEval, x)
        if type(xx) is dict:
            return tensor.softmax(list(xx.values())[0])
        else:
            return tensor.softmax(xx)

    def topo_sort(self, layers, src_of_layer):
        '''Topology sort of layers.

        It would try to preserve the orders of the input layers.

        Args:
            layers: a list of layers; the layers from the output of the same
                layer (e.g., slice layer) should be added by users in correct
                order; This function would not change their order.
            src_of_layer: a dictionary: src layer name -> a list of src layers

        Returns:
            A list of ordered layer
        '''
        order = []
        while len(order) < len(layers):
            for lyr in self.layers:
                if lyr not in order:
                    for src in src_of_layer[lyr.name]:
                        if src not in order:
                            break
                    order.append(lyr)
        return order

    def forward(self, flag, x, output=[], freeze=None):
        '''Forward the input(s) through every layer.

        Args:
            flag: True for training; False for evaluation; could also be
                model_pb2.kTrain or model_pb2.kEval, or other values for future
                use.
            x: a single SINGA tensor if there is a single input; otherwise, a
                dictionary: layer name-> singa tensor, for each layer accepting
                input data. Do not associate a layer with input tensor if it is
                connected from another layer. For such case, use a Dummy() layer
                to accept the input data and connect the dummy layer to this
                layer.
            output(list): a list of layer names whose output would be returned
                in addition to the default output.
            freeze(str): layer name, freeze all layers before this layer; flag
                is set to false for these layers.

        Returns:
            if there is only one output layer and output arg is empty, return
                the result from the single output layer; otherwise, return a
                dictionary: layer name -> output tensor(s)
        '''
        if self.ordered_layers is None:
            self.ordered_layers = self.topo_sort(
                self.layers, self.src_of_layer)
        if type(x) is dict:
            input_of_layer = x
        else:
            assert isinstance(x, tensor.Tensor), \
                'The inputs of a net should be dict or a single tensor'
            input_of_layer = {self.ordered_layers[0].name: x}
        output_of_layer = {}  # outputs generated by each layer
        ret = {}  # outputs to return
        if freeze is not None:
            is_valid = False
            for lyr in self.ordered_layers:
                is_valid |= lyr.name == freeze
            assert is_valid, 'Invalid freeze layer name =%s' % freeze
            old_flag = flag
            flag = False
        for cur in self.ordered_layers:
            if cur.name == freeze:
                flag = old_flag
            inputs = []
            if cur.name in input_of_layer:
                if type(input_of_layer[cur.name]) is list:
                    inputs.extend(input_of_layer[cur.name])
                else:
                    inputs.append(input_of_layer[cur.name])
            srcs = self.src_of_layer[cur.name]
            disp_src = ''
            for src in srcs:
                outs = output_of_layer[src.name]
                if type(outs) == list:
                    assert len(outs) > 0, \
                        'the output from layer %s is empty' % src.name
                    inputs.append(outs[0])
                    outs.pop(0)
                    if len(outs) == 0:
                        output_of_layer.pop(src.name)
                else:
                    inputs.append(outs)
                    output_of_layer[cur.name] = []
                    output_of_layer.pop(src.name)
            if len(inputs) == 1:
                inputs = inputs[0]
            out = cur.forward(flag, inputs)
            if verbose:
                disp_src = '+'.join([src.name for src in srcs])
                disp_src += '-->' + cur.name
                if type(out) is list:
                    print('%s: %s' % (disp_src,
                                      ' '.join([str(o.l1()) for o in out])))
                else:
                    print('%s: %f' % (disp_src, out.l1()))
            output_of_layer[cur.name] = out
            if cur.name in output:
                ret[cur.name] = out
            # print lyr.name, x.l1()
        # print output_of_layer
        ret.update(output_of_layer)
        if len(ret) == 1:
            return list(ret.values())[0]
        else:
            return ret

    def backward(self, dy, output=[], freeze=None):
        '''Run back-propagation after forward-propagation.

        Args:
            dy: a single tensor if there is a single loss function; otherwise,
                a dictionary maps the name of the layer connecting to the loss
                function -> gradient from the loss function. Do not associate a
                layer with gradient tensor if it is connecting to another layer.
                For such case, connect this layer to a Dummy() layer and use the
                dummy layer to accept the gradient.
            output(list): a list of layer names whose output gradient would be
                returned in addition to the param gradient
            freeze(str): layer name, stop backward after this layer.

        Returns:
                a geneartor iterator that generates
                (param_names, param_values, param_grads, layer_grads) after
                processing each layer h, where the first three lists are for h
                and the last item is a dictionary which maps
                layer name -> its output gradient tensor(s). At the end of this
                function, the key set includes all layers in the output arg.
        '''
        if self.dst_of_layer is None:
            self.dst_of_layer = {}
            for cur in self.layers:
                self.dst_of_layer[cur.name] = []
            for cur in self.ordered_layers[1:]:
                srcs = self.src_of_layer[cur.name]
                for src in srcs:
                    self.dst_of_layer[src.name].append(cur)
        output_of_layer = {}  # outputs generated by each layer
        ret = {}  # outputs to return
        if type(dy) is dict:
            input_of_layer = dy
        else:
            assert isinstance(dy, tensor.Tensor), \
                'The inputs of a net should be dict or a single tensor'
            input_of_layer = {self.ordered_layers[-1].name: dy}
        for cur in reversed(self.ordered_layers):
            inputs = []
            if cur.name in input_of_layer:
                if type(input_of_layer[cur.name]) is list:
                    inputs.extend(input_of_layer[cur.name])
                else:
                    inputs.append(input_of_layer[cur.name])
            for dst in self.dst_of_layer[cur.name]:
                outputs = output_of_layer[dst.name]
                if type(outputs) == list:
                    assert len(outputs) > 0, \
                        'the gradient from layer %s is empty' % dst.name
                    inputs.append(outputs[0])
                    outputs.pop(0)
                else:
                    inputs.append(outputs)
                    output_of_layer[dst.name] = []
                # del output_of_layer[dst.name]
            if len(inputs) == 1:
                inputs = inputs[0]
            outs, pgrads = cur.backward(kTrain, inputs)
            if verbose:
                disp_src = '+'.join(
                    [dst.name for dst in self.dst_of_layer[cur.name]])
                disp_src += '-->' + cur.name
                if type(outs) is list:
                    print('%s: %s' % (disp_src,
                                      ' '.join([str(o.l1()) for o in outs])))
                else:
                    print('%s: %f' % (disp_src, outs.l1()))
            if type(outs) is list:
                output_of_layer[cur.name] = outs[::-1]
            else:
                output_of_layer[cur.name] = outs
            if cur.name in output:
                ret[cur.name] = outs
            # ret.update(output_of_layer)
            yield (cur.param_names(), cur.param_values(), pgrads, ret)
            if cur.name == freeze:
                break

    def save(self, f, buffer_size=10, use_pickle=False):
        '''Save model parameters using io/snapshot.

        Args:
            f: file name
            buffer_size: size (MB) of the IO, default setting is 10MB; Please
                make sure it is larger than any single parameter object.
            use_pickle(Boolean): if true, it would use pickle for dumping;
                otherwise, it would use protobuf for serialization, which uses
                less space.
        '''
        if use_pickle:
            params = {}
            # since SINGA>=1.1.1  (1101)
            params['SINGA_VERSION'] = __version__
            for (name, val) in zip(self.param_names(), self.param_values()):
                val.to_host()
                params[name] = tensor.to_numpy(val)
            if not f.endswith('.pickle'):
                f = f + '.pickle'
            with open(f, 'wb') as fd:
                pickle.dump(params, fd)
        else:
            if f.endswith('.bin'):
                f = f[0:-4]
            sp = snapshot.Snapshot(f, True, buffer_size)
            v = tensor.from_numpy(np.array([__version__]))
            params['SINGA_VERSION'] = v
            for (name, val) in zip(self.param_names(), self.param_values()):
                val.to_host()
                sp.write(name, val)

    def load(self, f, buffer_size=10, use_pickle=False):
        '''Load model parameters using io/snapshot.

        Please refer to the argument description in save().
        '''
        version = 0

        def get_name(name):
            if version < 1101:
                idx = name.rfind('/')
                assert idx > 0, '/ must be in the parameter name'
                name = name[:idx] + '_' + name[idx + 1:]
            return name

        if use_pickle:
            print('NOTE: If your model was saved using Snapshot, '
                  'then set use_pickle=False for loading it')
            if not os.path.exists(f):
                # guess the correct path
                if f.endswith('.pickle'):
                    f = f[0:-7]
                else:
                    f = f + '.pickle'
            assert os.path.exists(f), 'file not exists %s w/o .pickle' % f
            with open(f, 'rb') as fd:
                params = pickle.load(fd, encoding='iso-8859-1')
        else:
            print('NOTE: If your model was saved using pickle, '
                  'then set use_pickle=True for loading it')
            if f.endswith('.bin'):
                f = f[0:-4]
            sp = snapshot.Snapshot(f, False, buffer_size)
            params = sp.read()

        if 'SINGA_VERSION' in params:
            version = params['SINGA_VERSION']
            if isinstance(version, tensor.Tensor):
                version = tensor.to_numpy(version)[0]
        else:
            version = 1100
        for name, val in zip(self.param_names(), self.param_values()):
            name = get_name(name)
            if name not in params:
                print('Param: %s missing in the checkpoint file' % name)
                continue
            try:
                if isinstance(params[name], tensor.Tensor):
                    val.copy_data(params[name])
                else:
                    val.copy_from_numpy(params[name])
            except AssertionError as err:
                print('Error from copying values for param: %s' % name)
                print(('shape of param vs checkpoint',
                       val.shape, params[name].shape))
                raise err
