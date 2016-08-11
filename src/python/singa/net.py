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
import cPickle as pickle


class FeedForwardNet(object):

    def __init__(self, loss=None, metric=None):
        self.loss = loss
        self.metric = metric
        self.layers = []

    def to_device(self, dev):
        for lyr in self.layers:
            lyr.to_device(dev)

    def add(self, lyr):
        """Append a layer into the layer list.

        This function will get the sample shape from the last layer to setup
        the newly added layer. For the first layer, it is setup outside.
        The calling function should ensure the correctness of the layer order.

        Args:
            lyr (Layer): the layer to be added
        """
        if len(self.layers) > 0 and lyr.has_setup is False:
            shape = self.layers[-1].get_output_sample_shape()
            print shape
            lyr.setup(shape)
        self.layers.append(lyr)

    def param_values(self):
        values = []
        for lyr in self.layers:
            values.extend(lyr.param_values())
        return values

    def param_specs(self):
        specs = []
        for lyr in self.layers:
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

    def forward(self, flag, x):
        # print x.l1()
        for lyr in self.layers:
            x = lyr.forward(flag, x)
        #    print lyr.name, x.l1()
        return x

    def backward(self):
        grad = self.loss.backward()
        pgrads = []
        for lyr in reversed(self.layers):
            grad, _pgrads = lyr.backward(kTrain, grad)
            for g in reversed(_pgrads):
                pgrads.append(g)
        return reversed(pgrads)

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
