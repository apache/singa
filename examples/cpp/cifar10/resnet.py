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
"""The resnet model is adapted from http://torch.ch/blog/2016/02/04/resnets.html
The best validation accuracy we achieved is about 83% without data augmentation.
The performance could be improved by tuning some hyper-parameters, including
learning rate, weight decay, max_epoch, parameter initialization, etc.
"""
from __future__ import print_function
from builtins import zip

from singa import layer
from singa import initializer
from singa import metric
from singa import loss
from singa import net as ffnet


def Block(net, name, nb_filters, stride):
    split = net.add(layer.Split(name + "-split", 2))
    if stride > 1:
        net.add(layer.Conv2D(name + "-br2-conv", nb_filters, 1, stride, pad=0), split)
        br2bn = net.add(layer.BatchNormalization(name + "-br2-bn"))
    net.add(layer.Conv2D(name + "-br1-conv1", nb_filters, 3, stride, pad=1), split)
    net.add(layer.BatchNormalization(name + "-br1-bn1"))
    net.add(layer.Activation(name + "-br1-relu"))
    net.add(layer.Conv2D(name + "-br1-conv2", nb_filters, 3, 1, pad=1))
    br1bn2 = net.add(layer.BatchNormalization(name + "-br1-bn2"))
    if stride > 1:
        net.add(layer.Merge(name + "-merge"), [br1bn2, br2bn])
    else:
        net.add(layer.Merge(name + "-merge"), [br1bn2, split])


def create_net(use_cpu=False):
    if use_cpu:
        layer.engine = 'singacpp'

    net = ffnet.FeedForwardNet(loss.SoftmaxCrossEntropy(), metric.Accuracy())
    net.add(layer.Conv2D("conv1", 16, 3, 1, pad=1, input_sample_shape=(3, 32, 32)))
    net.add(layer.BatchNormalization("bn1"))
    net.add(layer.Activation("relu1"))

    Block(net, "2a", 16, 1)
    Block(net, "2b", 16, 1)
    Block(net, "2c", 16, 1)

    Block(net, "3a", 32, 2)
    Block(net, "3b", 32, 1)
    Block(net, "3c", 32, 1)

    Block(net, "4a", 64, 2)
    Block(net, "4b", 64, 1)
    Block(net, "4c", 64, 1)

    net.add(layer.AvgPooling2D("pool4", 8, 8, border_mode='valid'))
    net.add(layer.Flatten('flat'))
    net.add(layer.Dense('ip5', 10))
    print('Start intialization............')
    for (p, name) in zip(net.param_values(), net.param_names()):
        # print name, p.shape
        if 'mean' in name or 'beta' in name:
            p.set_value(0.0)
        elif 'var' in name:
            p.set_value(1.0)
        elif 'gamma' in name:
            initializer.uniform(p, 0, 1)
        elif len(p.shape) > 1:
            if 'conv' in name:
                # initializer.gaussian(p, 0, math.sqrt(2.0/p.shape[1]))
                initializer.gaussian(p, 0, 9.0 * p.shape[0])
            else:
                initializer.uniform(p, p.shape[0], p.shape[1])
        else:
            p.set_value(0)
        # print name, p.l1()

    return net
