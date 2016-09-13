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

    def to_device(self, dev):
        for lyr in self.layers:
            lyr.to_device(dev)

    def add(self, lyr, src=None):
        """Append a layer into the layer list.

        This function will get the sample shape from the last layer to setup
        the newly added layer. For the first layer, it is setup outside.
        The calling function should ensure the correctness of the layer order.

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
            # print shape
            in_shape = self.src_of_layer[lyr.name][0].get_output_sample_shape()
            lyr.setup(in_shape)
            print lyr.name, lyr.get_output_sample_shape()
        self.layers.append(lyr)
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
        out = self.forward(kTrain, x)
        l = self.loss.forward(kTrain, out, y)
        if self.metric is not None:
            m = self.metric.evaluate(out, y)
        return self.backward(), (l.l1(), m)

    def evaluate(self, x, y):
        """Evaluate the loss and metric of the given data"""
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
        xx = self.forward(kEval, x)
        return tensor.softmax(xx)

    def topo_sort(self, cur, src_of_layer, visited=None, order=None):
        if visited is None:
            visited = {}
            for name in src_of_layer.keys():
                visited[name] = False
            order = []
        srcs = src_of_layer[cur.name]
        for src in srcs:
            if visited[src.name] is False:
                visited[src.name] = True
                self.topo_sort(src, src_of_layer, visited, order)
        order.append(cur)
        visited[cur.name] = True
        return order

    def forward(self, flag, x):
        # print x.l1()
        if self.ordered_layers is None:
            self.ordered_layers = self.topo_sort(self.layers[-1],
                                                 self.src_of_layer)
        inputs = [x]
        output_of_layer = {}
        for cur in self.ordered_layers:
            srcs = self.src_of_layer[cur.name]
            disp_src = cur.name + '<--'
            for src in srcs:
                outs = output_of_layer[src.name]
                if type(outs) == list:
                    assert len(outs) > 0, \
                            'the output from layer %s is empty' % src.name
                    inputs.append(outs[0])
                    outs.pop(0)
                else:
                    inputs.append(outs)
                    output_of_layer[cur.name] = []
                disp_src += '+' + src.name
                # del output_of_layer[src.name]
            # print disp_src
            if len(inputs) == 1:
                inputs = inputs[0]
            out= cur.forward(flag, inputs)
            if verbose:
                print '%s: %f' % (cur.name, out.l1())
            output_of_layer[cur.name] = out
            inputs = []
            # print lyr.name, x.l1()
        # print output_of_layer
        return output_of_layer[self.ordered_layers[-1].name]

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

    def save(self, f):
        """Save model parameters using cpickle"""
        params = {}
        for (specs, val) in zip(self.param_specs(), self.param_values()):
            val.to_host()
            params[specs.name] = tensor.to_numpy(val)
        with open(f, 'wb') as fd:
            pickle.dump(params, fd)

    def load(self, f):
        """Load model parameters using cpickle"""
        with open(f, 'rb') as fd:
            params = pickle.load(fd)
        for (specs, val) in zip(self.param_specs(), self.param_values()):
            val.copy_from_numpy(params[specs.name])
