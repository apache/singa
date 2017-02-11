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
''' This models are created following https://github.com/facebook/fb.resnet.torch.git
and https://github.com/szagoruyko/wide-residual-networks
'''
from singa.layer import Conv2D, Activation, MaxPooling2D, AvgPooling2D,\
        Split, Merge, Flatten, Dense, BatchNormalization, Softmax
from singa import net as ffnet
from singa import initializer

ffnet.verbose=True

conv_bias = False

def conv(net, prefix, n, ksize, stride=1, pad=0, bn=True, relu=True, src=None):
    '''Add a convolution layer and optionally a batchnorm and relu layer.

    Args:
        prefix, a string for the prefix of the layer name
        n, num of filters for the conv layer
        bn, if true add batchnorm
        relu, if true add relu

    Returns:
        the last added layer
    '''
    ret = net.add(Conv2D(
        prefix + '-conv', n, ksize, stride, pad=pad, use_bias=conv_bias), src)
    if bn:
        ret = net.add(BatchNormalization(prefix + '-bn'))
    if relu:
        ret = net.add(Activation(prefix + '-relu'))
    return ret


def shortcut(net, prefix, inplane, outplane, stride, src, bn=False):
    '''Add a conv shortcut layer if inplane != outplane; or return the source
    layer directly.

    Args:
        prefix, a string for the prefix of the layer name
        bn, if true add a batchnorm layer after the conv layer

    Returns:
        return the last added layer or the source layer.
    '''
    if inplane == outplane:
        return src
    return conv(net, prefix + '-shortcut', outplane, 1, stride, 0, bn, False, src)


def bottleneck(name, net, inplane, midplane, outplane, stride=1, preact=False, add_bn=False):
    '''Add three conv layers, with a>=b<=c filters.

    The default structure is
    input
         -split - conv1-bn1-relu1-conv2-bn2-relu2-conv3-bn3
                - conv-bn or dummy
         -add
         -relu

    Args:
        inplane, num of feature maps of the input
        midplane, num of featue maps of the middle layer
        outplane, num of feature maps of the output
        preact, if true, move the bn3 and relu before conv1, i.e., pre-activation ref identity mapping paper
        add_bn, if true, move the last bn after the addition layer (for resnet-50)
    '''
    assert not (preact and add_bn), 'preact and batchnorm after addition cannot be true at the same time'
    split = net.add(Split(name + '-split', 2))
    if preact:
        net.add(BatchNormalization(name + '-preact-bn'))
        net.add(Activation(name + '-preact-relu'))
    conv(net, name + '-0', midplane, 1, 1, 0, True, True)
    conv(net, name + '-1', midplane, 3, stride, 1, True, True)
    br0 = conv(net, name + '-2', outplane, 1, 1, 0, not (preact or add_bn), False)
    br1 = shortcut(net, name, inplane, outplane, stride, split, not add_bn)
    ret = net.add(Merge(name + '-add'), [br0, br1])
    if add_bn:
        ret = net.add(BatchNormalization(name + '-add-bn'))
    if not preact:
        ret = net.add(Activation(name + '-add-relu'))
    return ret


def basicblock(name, net, inplane, midplane, outplane, stride=1, preact=False, add_bn=False):
    '''Add two conv layers, with a<=b filters.

    The default structure is
    input
         -split - conv1-bn1-relu1-conv2-bn2
                - conv or dummy
         -add
         -relu

    Args:
        inplane, num of feature maps of the input
        midplane, num of featue maps of the middle layer
        outplane, num of feature maps of the output
        preact, if true, move the bn2 and relu before conv1, i.e., pre-activation ref identity mapping paper
        add_bn, if true, move the last bn after the addition layer (for resnet-50)
    '''
    assert not (preact and add_bn), 'preact and batchnorm after addition cannot be true at the same time'
    split = net.add(Split(name + '-split', 2))
    if preact:
        net.add(BatchNormalization(name + '-preact-bn'))
        net.add(Activation(name + '-preact-relu'))
    conv(net, name + '-0', midplane, 3, stride, 1, True, True)
    br0 = conv(net, name + '-1', outplane, 3, 1, 1, not preact, False)
    br1 = shortcut(net, name, inplane, outplane, stride, split, False)
    ret = net.add(Merge(name + '-add'), [br0, br1])
    if add_bn:
        ret = net.add(BatchNormalization(name + '-add-bn'))
    if not preact:
        ret = net.add(Activation(name + '-add-relu'))
    return ret


def stage(sid, net, num_blk, inplane, midplane, outplane, stride, block, preact=False, add_bn=False):
    block('stage%d-blk%d' % (sid, 0), net, inplane, midplane, outplane, stride, preact, add_bn)
    for i in range(1, num_blk):
        block('stage%d-blk%d' % (sid, i), net, outplane, midplane, outplane, 1, preact, add_bn)

def init_params(net, weight_path=None):
    if weight_path == None:
        for pname, pval in zip(net.param_names(), net.param_values()):
            print pname, pval.shape
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
        net.load(weight_path, use_pickle = 'pickle' in weight_path)


cfg = { 18: [2, 2, 2, 2],  # basicblock
        34: [3, 4, 6, 3],  # basicblock
        50: [3, 4, 6, 3],  # bottleneck
        101: [3, 4, 23, 3], # bottleneck
        152: [3, 8, 36, 3], # bottleneck
        200: [3, 24, 36, 3]} # bottleneck


def create_addbn_resnet(depth=50):
    '''Original resnet with the last batchnorm of each block moved to after the addition layer'''
    net = ffnet.FeedForwardNet()
    net.add(Conv2D('input-conv', 64, 7, 2, pad=3, use_bias=False, input_sample_shape=(3, 224, 224)))
    net.add(BatchNormalization('input-bn'))
    net.add(Activation('input_relu'))
    net.add(MaxPooling2D('input_pool', 3, 2, pad=1))
    conf = cfg[depth]
    if depth > 34:
        stage(0, net, conf[0], 64, 64, 256, 1, bottleneck, add_bn=True)
        stage(1, net, conf[1], 256, 128, 512, 2, bottleneck, add_bn=True)
        stage(2, net, conf[2], 512, 256, 1024, 2, bottleneck, add_bn=True)
        stage(3, net, conf[3], 1024, 512, 2048, 2, bottleneck, add_bn=True)
    else:
        stage(0, net, conf[0], 64, 64, 64, 1, basicblock, add_bn=True)
        stage(1, net, conf[1], 64, 128, 128, 2, basicblock, add_bn=True)
        stage(2, net, conf[2], 128, 256, 256, 2, basicblock, add_bn=True)
        stage(3, net, conf[3], 256, 512, 512, 2, basicblock, add_bn=True)
    net.add(AvgPooling2D('avg', 7, 1, pad=0))
    net.add(Flatten('flat'))
    net.add(Dense('dense', 1000))
    return net


def create_resnet(depth=18):
    '''Original resnet, where the there is a relue after the addition layer'''
    net = ffnet.FeedForwardNet()
    net.add(Conv2D('input-conv', 64, 7, 2, pad=3, use_bias=False, input_sample_shape=(3, 224, 224)))
    net.add(BatchNormalization('input-bn'))
    net.add(Activation('input_relu'))
    net.add(MaxPooling2D('input_pool', 3, 2, pad=1))
    conf = cfg[depth]
    if depth > 34:
        stage(0, net, conf[0], 64, 64, 256, 1, bottleneck)
        stage(1, net, conf[1], 256, 128, 512, 2, bottleneck)
        stage(2, net, conf[2], 512, 256, 1024, 2, bottleneck)
        stage(3, net, conf[3], 1024, 512, 2048, 2, bottleneck)
    else:
        stage(0, net, conf[0], 64, 64, 64, 1, basicblock)
        stage(1, net, conf[1], 64, 128, 128, 2, basicblock)
        stage(2, net, conf[2], 128, 256, 256, 2, basicblock)
        stage(3, net, conf[3], 256, 512, 512, 2, basicblock)
    net.add(AvgPooling2D('avg', 7, 1, pad=0))
    net.add(Flatten('flat'))
    net.add(Dense('dense', 1000))
    return net

def create_preact_resnet(depth=200):
    '''Resnet with the batchnorm and relu moved to before the conv layer for each block'''
    net = ffnet.FeedForwardNet()
    net.add(Conv2D('input-conv', 64, 7, 2, pad=3, use_bias=False, input_sample_shape=(3, 224, 224)))
    net.add(BatchNormalization('input-bn'))
    net.add(Activation('input_relu'))
    net.add(MaxPooling2D('input_pool', 3, 2, pad=1))
    conf = cfg[depth]
    if depth > 34:
        stage(0, net, conf[0], 64, 64, 256, 1, bottleneck, preact=True)
        stage(1, net, conf[1], 256, 128, 512, 2, bottleneck, preact=True)
        stage(2, net, conf[2], 512, 256, 1024, 2, bottleneck, preact=True)
        stage(3, net, conf[3], 1024, 512, 2048, 2, bottleneck, preact=True)
    else:
        stage(0, net, conf[0], 64, 64, 64, 1, basicblock, preact=True)
        stage(1, net, conf[1], 64, 128, 128, 2, basicblock, preact=True)
        stage(2, net, conf[2], 128, 256, 256, 2, basicblock, preact=True)
        stage(3, net, conf[3], 256, 512, 512, 2, basicblock, preact=True)
    net.add(BatchNormalization('final-bn'))
    net.add(Activation('final-relu'))
    net.add(AvgPooling2D('avg', 7, 1, pad=0))
    net.add(Flatten('flat'))
    net.add(Dense('dense', 1000))
    return net


def create_wide_resnet(depth=50):
    '''Similar original resnet except that a<=b<=c for the bottleneck block'''
    net = ffnet.FeedForwardNet()
    net.add(Conv2D('input-conv', 64, 7, 2, pad=3, use_bias=False, input_sample_shape=(3, 224, 224)))
    net.add(BatchNormalization('input-bn'))
    net.add(Activation('input_relu'))
    net.add(MaxPooling2D('input_pool', 3, 2, pad=1))

    stage(0, net, 3, 64, 128, 256, 1, bottleneck)
    stage(1, net, 4, 256, 256, 512, 2, bottleneck)
    stage(2, net, 6, 512, 512, 1024, 2, bottleneck)
    stage(3, net, 3, 1024, 1024, 2048, 2, bottleneck)

    net.add(AvgPooling2D('avg_pool', 7, 1, pad=0))
    net.add(Flatten('flag'))
    net.add(Dense('dense', 1000))
    return net


def create_net(name, depth):
    if name == 'resnet':
        return create_resnet(depth)
    elif name == 'wrn':
        return create_wide_resnet(depth)
    elif name == 'preact':
        return create_preact_resnet(depth)
    elif name == 'addbn':
        return create_addbn_resnet(depth)


if __name__ == '__main__':
    create_net('wrn', 50)
