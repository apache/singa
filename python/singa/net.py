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
"""


from .proto.model_pb2 import kTrain, kEval
import tensor
import layer
import snapshot
import cPickle as pickle

'''For display training information, e.g L1 value of layer data'''
verbose = False


class FeedForwardNet(object):

    def __init__(self, loss=None, metric=None):
        self.loss = loss
        self.metric = metric
        self.layers = []
        self.src_of_layer = {}
        self.dst_of_layer = None
        self.ordered_layers = None
        self.out_sample_shape_of_layer = {}

    def to_device(self, dev):
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
        print lyr.name, out_shape
        return lyr

    def param_values(self):
        values = []
        layers = self.layers
        if self.ordered_layers is not None:
            layers = self.ordered_layers
        for lyr in layers:
            values.extend(lyr.param_values())
        return values

    def param_specs(self):
        specs = []
        layers = self.layers
        if self.ordered_layers is not None:
            layers = self.ordered_layers
        for lyr in layers:
            specs.extend(lyr.param_specs)
        return specs

    def param_names(self):
        return [spec.name for spec in self.param_specs()]

    def train(self, x, y):
        '''Run BP for one iteration.

        Currently only support nets with a single output layer, and a single
        loss objective and metric.
        TODO(wangwei) consider multiple loss objectives and metrics.

        Args:
            x: input data, a single input Tensor or a dict: layer name -> Tensor
            y: label data, a single input Tensor.

        Returns:
            gradients of parameters and the loss and metric values.
        '''
        out = self.forward(kTrain, x)
        l = self.loss.forward(kTrain, out, y)
        if self.metric is not None:
            m = self.metric.evaluate(out, y)
        return self.backward(), (l.l1(), m)

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

        Args:
            x: input data, a single input Tensor or a dict: layer name -> Tensor

        Returns:
            a single output tensor as the prediction result.
        '''
        xx = self.forward(kEval, x)
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

    def forward(self, flag, x, output=[]):
        '''Forward the input(s) through every layer.

        If a layer has inputs from other layers and from x, the data from x is
        ordered before the data from other layers, e.g., if layer 1 -> layer 2,
        and x['layer 2'] has data, then the input of layer 2 is
        flatten([x['layer 2'], output of layer 1])

        Args:
            flag: True for training; False for evaluation; could also be
                model_pb2.kTrain or model_pb2.kEval, or other values for future
                use.
            x: a single SINGA tensor or a dictionary: layer name-> singa tensor
            output(list): a list of layer names whose output would be returned
                in addition to the default output

        Returns:
            if there is only one output layer, return its output tensor(s);
            else return a dictionary: layer name -> output tensor(s)
        '''
        if self.ordered_layers is None:
            self.ordered_layers = self.topo_sort(self.layers, self.src_of_layer)
        if type(x) is dict:
            input_of_layer = x
        else:
            assert isinstance(x, tensor.Tensor), \
                'The inputs of a net should be dict or a single tensor'
            input_of_layer = {self.ordered_layers[0].name: x}
        output_of_layer = {}  # outputs generated by each layer
        ret = {}  # outputs to return
        for cur in self.ordered_layers:
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
                    print '%s: %s' % (disp_src,
                                      ' '.join([str(o.l1()) for o in out]))
                else:
                    print '%s: %f' % (disp_src, out.l1())
            output_of_layer[cur.name] = out
            if cur.name in output:
                ret[cur.name] = out
            # print lyr.name, x.l1()
        # print output_of_layer
        ret.update(output_of_layer)
        if len(ret) == 1:
            return ret.values()[0]
        else:
            return ret

    def backward(self):
        if self.dst_of_layer is None:
            self.dst_of_layer = {}
            for cur in self.layers:
                self.dst_of_layer[cur.name] = []
            for cur in self.ordered_layers[1:]:
                srcs = self.src_of_layer[cur.name]
                for src in srcs:
                    self.dst_of_layer[src.name].append(cur)
        grad = self.loss.backward()
        if len(grad.shape) > 1:
            grad /= grad.shape[0]  # average across the batch
        # print 'grad', grad.l1()
        grads = [grad]
        output_of_layer = {}
        pgrads = []
        for cur in reversed(self.ordered_layers):
            for dst in self.dst_of_layer[cur.name]:
                outputs = output_of_layer[dst.name]
                if type(outputs) == list:
                    assert len(outputs) > 0, \
                            'the gradient from layer %s is empty' % dst.name
                    grads.append(outputs[0])
                    outputs.pop(0)
                else:
                    grads.append(outputs)
                    output_of_layer[dst.name] = []
                # del output_of_layer[dst.name]
            if len(grads) == 1:
                grads = grads[0]
            outs, _pgrads = cur.backward(kTrain, grads)
            pgrads.append(_pgrads)
            output_of_layer[cur.name] = outs
            grads = []

        ret = []
        for pgrad in reversed(pgrads):
            ret.extend(pgrad)
        return ret

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
            for (specs, val) in zip(self.param_specs(), self.param_values()):
                val.to_host()
                params[specs.name] = tensor.to_numpy(val)
                with open(f, 'wb') as fd:
                    pickle.dump(params, fd)
        else:
            sp = snapshot.Snapshot(f, True, buffer_size)
            for (specs, val) in zip(self.param_specs(), self.param_values()):
                val.to_host()
                sp.write(specs.name, val)

    def load(self, f, buffer_size=10, use_pickle=False):
        '''Load model parameters using io/snapshot.

        Please refer to the argument description in save().
        '''
        if use_pickle:
            print 'NOTE: If your model was saved using Snapshot, '\
                    'then set use_pickle=False for loading it'
            with open(f, 'rb') as fd:
                params = pickle.load(fd)
                for (specs, val) in zip(self.param_specs(),
                                        self.param_values()):
                    val.copy_from_numpy(params[specs.name])
        else:
            print 'NOTE: If your model was saved using pickle, '\
                    'then set use_pickle=True for loading it'
            sp = snapshot.Snapshot(f, False, buffer_size)
            params = sp.read()
            for (specs, val) in zip(self.param_specs(), self.param_values()):
                val.copy_data(params[specs.name])
