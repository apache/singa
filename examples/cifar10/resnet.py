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
""" The resnet model is adapted from http://torch.ch/blog/2016/02/04/resnets.html
The best validation accuracy we achieved is about 83% without data augmentation.
The performance could be improved by tuning some hyper-parameters, including
learning rate, weight decay, max_epoch, parameter initialization, etc.
"""

import sys
import os
import math
import cPickle as pickle

#sys.path.append(os.path.join(os.path.dirname(__file__), '../../build/python'))
# use the python modules by installing py singa in build/python
# pip install -e .

from singa import tensor
from singa import layer
from singa import initializer
from singa import metric
from singa import loss
from singa import net as ffnet
from singa.proto.model_pb2 import kTrain, kEval

class ResNet(object):

    def __init__(self, loss=None, metric=None):
        self.loss = loss
        self.metric = metric
        self.layers = []
        self.src_layers = {}
        self.dst_layers = {}
        self.layer_shapes = {}
        self.layer_names = []

    def to_device(self, dev):
        for lyr in self.layers:
            lyr.to_device(dev)

    def find(self, name):
        for i in xrange(len(self.layers)):
            if self.layers[i].name == name:
                return self.layers[i]
        assert False, "Undefined layer %s." % name
        return None

    def add(self, lyr, src_lyr_name=''):
        """Append a layer into the layer list.
        This function will get the sample shape from the last layer to setup
        the newly added layer. For the first layer, it is setup outside.
        The calling function should ensure the correctness of the layer order.
        Args:
            lyr (Layer): the layer to be added
            src_lyr_name: list type, name of the src layer to the current layer
        """
        if len(self.layers) > 0 and lyr.has_setup is False:
            #assert src_lyr_name in dst_layers, "Undefined src layer %s" % src_lyr_name
            shape = self.layer_shapes[src_lyr_name]
            lyr.setup(shape)
        print lyr.name, ': ', lyr.get_output_sample_shape()
        if src_lyr_name != '':
            self.src_layers[lyr.name] = [src_lyr_name]
        self.layers.append(lyr)
        self.layer_shapes[lyr.name] = lyr.get_output_sample_shape()            
        self.layer_names.append(lyr.name)

        if src_lyr_name != '':
            if src_lyr_name in self.dst_layers:
                self.dst_layers[src_lyr_name].append(lyr.name)
            else:
                self.dst_layers[src_lyr_name] = [lyr.name]
        if lyr.name in self.src_layers:
            print 'src: ', self.src_layers[lyr.name]
        else:
            print 'src: null'
        #print self.layer_names
        print "----------------------------------------"

    def add_split(self, lyr_name, src_lyr_name):
        assert src_lyr_name in self.layer_shapes, "Undefined src layer %s." % src_lyr_name
        self.src_layers[lyr_name] = [src_lyr_name]
        self.layer_shapes[lyr_name] = self.layer_shapes[src_lyr_name]
        self.layer_names.append(lyr_name)
        if src_lyr_name in self.dst_layers:
            self.dst_layers[src_lyr_name].append(lyr_name)
        else:
            self.dst_layers[src_lyr_name] = [lyr_name]
        print lyr_name, ': ', self.layer_shapes[lyr_name]
        if lyr_name in self.src_layers:
            print 'src: ', self.src_layers[lyr_name]
        else:
            print 'src: null'
        print "----------------------------------------"
   
    def add_merge(self, lyr_name, src_lyr_names):
        self.src_layers[lyr_name] = src_lyr_names
        self.layer_shapes[lyr_name] = self.layer_shapes[src_lyr_names[0]]
        self.layer_names.append(lyr_name)
        for i in xrange(len(src_lyr_names)):
            if src_lyr_names[i] in self.dst_layers:
                self.dst_layers[src_lyr_names[i]].append(lyr_name)
            else:
                self.dst_layers[src_lyr_names[i]] = [lyr_name]
        print lyr_name, ': ', self.layer_shapes[lyr_name]
        if lyr_name in self.src_layers:
            print 'src: ', self.src_layers[lyr_name]
        else:
            print 'src: null'
        print "----------------------------------------"

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
        #print x.l1()
        outputs = {'': x}
        for idx, name in enumerate(self.layer_names):
            #print 'forward layer', name
            if idx == 0:
                outputs[name] = self.find(name).forward(flag, outputs[''])
                del outputs['']
                continue

            if 'split' in name:
                src = self.src_layers[name][0]
                #print 'src: ', src
                outputs[name] = []
                for i in xrange(len(self.dst_layers[name])):
                    outputs[name].append(outputs[src])
                del outputs[src]
            elif 'merge' in name:
                srcs = self.src_layers[name]
                #print 'src: ', srcs
                for i in xrange(len(srcs)):
                    if 'split' in srcs[i]:
                       if i > 0:
                            data += outputs[srcs[i]][0]
                       else:
                            data = outputs[srcs[i]][0]
                       del outputs[srcs[i]][0]
                       if len(outputs[srcs[i]]) == 0:
                           del outputs[srcs[i]]
                    else:
                        if i > 0:
                            data += outputs[srcs[i]]
                        else:
                            data = outputs[srcs[i]]
                        del outputs[srcs[i]]
                outputs[name] = data
            else:
                src = self.src_layers[name][0]
                #print 'src: ', src
                if 'split' in src:
                    outputs[name] = self.find(name).forward(flag, outputs[src][0])
                    del outputs[src][0]
                    if len(outputs[src]) == 0:
                        del outputs[src]
                else:
                    outputs[name] = self.find(name).forward(flag, outputs[src])
                    del outputs[src]
                
        #    print lyr.name, x.l1()
        return outputs[name]

    def backward(self, flag=kTrain):
        grad = self.loss.backward()
        pgrads = []
        in_grads = {'': grad}
        for idx, name in enumerate(reversed(self.layer_names)):
            #print 'backward layer', name
            if idx == 0:
                lyr = self.find(name)
                grad, _pgrads = lyr.backward(flag, in_grads[''])
                for g in reversed(_pgrads):
                    pgrads.append(g)
                in_grads[name] = grad
                del in_grads['']
                continue

            if 'merge' in name:
                src = self.dst_layers[name][0]
                #print 'src: ', src
                in_grads[name] = []
                for i in xrange(len(self.src_layers[name])):
                    in_grads[name].append(in_grads[src])
                del in_grads[src]
            elif 'split' in name:
                srcs = self.dst_layers[name]
                #print 'src: ', srcs
                for i in xrange(len(srcs)):
                    if 'merge' in srcs[i]:
                       if i > 0:
                            data += in_grads[srcs[i]][0]
                       else:
                            data = in_grads[srcs[i]][0]
                       del in_grads[srcs[i]][0]
                       if len(in_grads[srcs[i]]) == 0:
                           del in_grads[srcs[i]]
                    else:
                        if i > 0:
                            data += in_grads[srcs[i]]
                        else:
                            data = in_grads[srcs[i]]
                        del in_grads[srcs[i]]
                in_grads[name] = data
            else:
                src = self.dst_layers[name][0]
                #print 'src: ', src
                if 'merge' in src:
                    grad, _pgrads = self.find(name).backward(flag, in_grads[src][0])
                    del in_grads[src][0]
                    if len(in_grads[src]) == 0:
                        del in_grads[src]
                else:
                    grad, _pgrads = self.find(name).backward(flag, in_grads[src])
                    del in_grads[src]
                for g in reversed(_pgrads):
                    pgrads.append(g)
                in_grads[name] = grad


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

def Block(net, name, nb_filters, stride, std, src):
    #net.add(layer.Split("split" + name, 2), srcs)
    net.add_split("split" + name, src)
    if stride > 1:
        net.add(layer.Conv2D("conv" + name + "_br1", nb_filters, 1, stride, pad=0), "split" + name)
        net.add(layer.BatchNormalization("bn" + name + "_br1"), "conv" + name + "_br1")
        net.add(layer.Conv2D("conv" + name + "_br2a", nb_filters, 3, stride, pad=1), "split" + name)
    else:
        net.add(layer.Conv2D("conv" + name + "_br2a", nb_filters, 3, stride, pad=1), "split" + name)
    net.add(layer.BatchNormalization("bn" + name + "_br2a"), "conv" + name + "_br2a")
    net.add(layer.Activation("relu" + name + "_br2a"), "bn" + name + "_br2a")
    net.add(layer.Conv2D("conv" + name + "_br2b", nb_filters, 3, 1, pad=1), "relu" + name + "_br2a")
    net.add(layer.BatchNormalization("bn" + name + "_br2b"), "conv" + name + "_br2b")
    if stride > 1:
        net.add_merge("merge" + name, ["bn" + name + "_br1", "bn" + name + "_br2b"])
    else:
        net.add_merge("merge" + name, ["split" + name, "bn" + name + "_br2b"])

def create_net():
    net = ResNet(loss.SoftmaxCrossEntropy(), metric.Accuracy())
    net.add(layer.Conv2D("conv1", 16, 3, 1, pad=1, input_sample_shape=(3, 32, 32)))
    net.add(layer.BatchNormalization("bn1"), "conv1")
    net.add(layer.Activation("relu1"), "bn1")
   
    Block(net, "2a", 16, 1, 0.01, "relu1")
    Block(net, "2b", 16, 1, 0.01, "merge2a")
    Block(net, "2c", 16, 1, 0.01, "merge2b")

    Block(net, "3a", 32, 2, 0.01, "merge2c")
    Block(net, "3b", 32, 1, 0.01, "merge3a")
    Block(net, "3c", 32, 1, 0.01, "merge3b")

    Block(net, "4a", 64, 2, 0.01, "merge3c")
    Block(net, "4b", 64, 1, 0.01, "merge4a")
    Block(net, "4c", 64, 1, 0.01, "merge4b")

    net.add(layer.AvgPooling2D("pool4", 8, 8, border_mode='valid'), "merge4c")
    net.add(layer.Flatten('flat'), "pool4")
    net.add(layer.Dense('ip5', 10), "flat")
    net.add(layer.Softmax('softmax'), "ip5")
    print 'Start intialization............'
    for (p, name) in zip(net.param_values(), net.param_names()):
        print name, p.shape
        if 'mean' in name or 'beta' in name:
            p.set_value(0.0)
        elif 'var' in name:
            p.set_value(1.0)
        elif 'gamma' in name:
            initializer.uniform(p, 0, 1)
        elif len(p.shape) > 1:
            if 'conv' in name:
                #initializer.gaussian(p, 0, math.sqrt(2.0/p.shape[1]))
                initializer.gaussian(p, 0, math.sqrt(2.0/(9.0*p.shape[0])))
            else:
                initializer.gaussian(p, 0, 0.02)
        else:
            p.set_value(0)
        print name, p.l1()

    return net
