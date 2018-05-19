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
''' This models are created following
https://arxiv.org/pdf/1608.06993.pdf and
https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
'''
from singa import initializer
from singa import layer
from singa import net as ffnet
from singa import loss
from singa import metric
from singa.layer import Conv2D, Activation, MaxPooling2D,\
    AvgPooling2D, Split, Concat, Flatten, BatchNormalization

import math
import sys

ffnet.verbose = True

conv_bias = False


def add_dense_connected_layers(name, net, growth_rate):
    net.add(BatchNormalization('%s/bn1' % name))
    net.add(Activation('%s/relu1' % name))
    net.add(Conv2D('%s/conv1' % name, 4 * growth_rate, 1, 1,
                   pad=0, use_bias=conv_bias))
    net.add(BatchNormalization('%s/bn2' % name))
    net.add(Activation('%s/relu2' % name))
    return net.add(Conv2D('%s/conv2' % name, growth_rate, 3, 1,
                          pad=1, use_bias=conv_bias))


def add_layer(name, net, growth_rate):
    split = net.add(Split('%s/split' % name, 2))
    dense = add_dense_connected_layers(name, net, growth_rate)
    net.add(Concat('%s/concat' % name, 1), [split, dense])


def add_transition(name, net, n_channels, last=False):
    net.add(BatchNormalization('%s/norm' % name))
    lyr = net.add(Activation('%s/relu' % name))
    if last:
        net.add(AvgPooling2D('%s/pool' % name, 
                             lyr.get_output_sample_shape()[1:3], pad=0))
        net.add(Flatten('flat'))
    else:
        net.add(Conv2D('%s/conv' % name, n_channels, 1, 1, 
                       pad=0, use_bias=conv_bias))
        net.add(AvgPooling2D('%s/pool' % name, 2, 2, pad=0))


def add_block(name, net, n_channels, N, growth_rate):
    for i in range(N):
        add_layer('%s/%d' % (name, i), net, growth_rate)
        n_channels += growth_rate
    return n_channels


def densenet_base(depth, growth_rate=32, reduction=0.5):
    '''
        rewrite according to pytorch models
        special case of densenet 161
    '''
    if depth == 121:
        stages = [6, 12, 24, 16]
    elif depth == 169:
        stages = [6, 12, 32, 32]
    elif depth == 201:
        stages = [6, 12, 48, 32]
    elif depth == 161:
        stages = [6, 12, 36, 24]
    else:
        print('unknown depth: %d' % depth)
        sys.exit(-1)

    net = ffnet.FeedForwardNet()
    growth_rate = 48 if depth == 161 else 32
    n_channels = 2 * growth_rate

    net.add(Conv2D('input/conv', n_channels, 7, 2, pad=3, 
                   use_bias=conv_bias, input_sample_shape=(3, 224, 224)))
    net.add(BatchNormalization('input/bn'))
    net.add(Activation('input/relu'))
    net.add(MaxPooling2D('input/pool', 3, 2, pad=1))

    # Dense-Block 1 and transition (56x56)
    n_channels = add_block('block1', net, n_channels, stages[0], growth_rate)
    add_transition('trans1', net, int(math.floor(n_channels*reduction)))
    n_channels = math.floor(n_channels*reduction)

    # Dense-Block 2 and transition (28x28)
    n_channels = add_block('block2', net, n_channels, stages[1], growth_rate)
    add_transition('trans2', net, int(math.floor(n_channels*reduction)))
    n_channels = math.floor(n_channels*reduction)

    # Dense-Block 3 and transition (14x14)
    n_channels = add_block('block3', net, n_channels, stages[2], growth_rate)
    add_transition('trans3', net, int(math.floor(n_channels*reduction)))
    n_channels = math.floor(n_channels*reduction)

    # Dense-Block 4 and transition (7x7)
    n_channels = add_block('block4', net, n_channels, stages[3], growth_rate)
    add_transition('trans4', net, n_channels, True)

    return net


def init_params(net, weight_path=None, is_train=False):
    '''Init parameters randomly or from checkpoint file.

        Args:
            net, a constructed neural net
            weight_path, checkpoint file path
            is_train, if false, then a checkpoint file must be presented
    '''
    assert is_train is True or weight_path is not None, \
        'must provide a checkpoint file for serving'

    if weight_path is None:
        for pname, pval in zip(net.param_names(), net.param_values()):
            if 'conv' in pname and len(pval.shape) > 1:
                initializer.gaussian(pval, 0, pval.shape[1])
            elif 'dense' in pname:
                if len(pval.shape) > 1:
                    initializer.gaussian(pval, 0, pval.shape[0])
                else:
                    pval.set_value(0)
            # init params from batch norm layer
            elif 'mean' in pname or 'beta' in pname:
                pval.set_value(0)
            elif 'var' in pname:
                pval.set_value(1)
            elif 'gamma' in pname:
                initializer.uniform(pval, 0, 1)
    else:
        net.load(weight_path, use_pickle=True)


def create_net(depth, nb_classes, dense=0, use_cpu=True):
    if use_cpu:
        layer.engine = 'singacpp'

    net = densenet_base(depth)

    # this part was not included in the pytorch model
    if dense > 0:
        net.add(layer.Dense('hidden-dense', dense))
        net.add(layer.Activation('act-dense'))
        net.add(layer.Dropout('dropout'))

    net.add(layer.Dense('sigmoid', nb_classes))
    return net

if __name__ == '__main__':
    create_net(121, 1000)