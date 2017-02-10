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
''' This model is created following https://github.com/facebook/fb.resnet.torch.git
'''
from singa.layer import Conv2D, Activation, MaxPooling2D, AvgPooling2D,\
        Split, Merge, Flatten, Dense, BatchNormalization, Softmax
from singa import net as ffnet
from singa import initializer

ffnet.verbose=True

conv_bias = False

def conv(net, prefix, n, ksize, stride=1, pad=0, bn=True, relu=True, src=None):
    ret = net.add(Conv2D(
        prefix + '-conv', n, ksize, stride, pad=pad, use_bias=conv_bias), src)
    if bn:
        ret = net.add(BatchNormalization(prefix + '-bn'))
    if relu:
        ret = net.add(Activation(prefix + '-relu'))
    return ret


def shortcut(net, prefix, inplane, outplane, stride, src):
    if inplane == outplane:
        return src
    return conv(net, prefix + '-shortcut', outplane, 1, stride, 0, True, False, src)


def bottleneck(name, net, inplane, midplane, outplane, stride=1, preact=False):
    split = net.add(Split(name + '-split', 2))
    conv(net, name + '-1', midplane, 1, 1, 0, True, True, src=split)
    conv(net, name + '-2', midplane, 3, stride, 1, True, True)
    br0 = conv(net, name + '-3', outplane, 1, 1, 0, True, False)
    br1 = shortcut(net, name, inplane, outplane, stride, split)
    net.add(Merge(name + '-add'), [br0, br1])
    return net.add(Activation(name + '-relu'))

def basicblock(name, net, inplane, midplane, outplane, stride=1, preact=False):
    assert midplane==outplane, 'midplan and outplane should be the same'
    split = net.add(Split(name + '-split', 2))
    if preact:
        net.add(BatchNormalization(name + '-preact-bn'), split)
        net.add(Activation(name + '-preact-relu'))
    conv(net, name + '-1', outplane, 3, stride, 1, True, True, split)
    br0 = conv(net, name + '-2', outplane, 3, 1, 1, True, False)
    br1 = shortcut(net, name, inplane, outplane, stride, split)
    net.add(Merge(name + '-add'), [br0, br1])
    return net.add(Activation(name + '-add-relu'))


def stage(sid, net, num_blk, inplane, midplane, outplane, stride, block):
    block('stage%d-blk%d' % (sid, 0), net, inplane, midplane, outplane, stride)
    for i in range(1, num_blk):
        block('stage%d-blk%d' % (sid, i), net, outplane, midplane, outplane)

def init_params(net, weight_path):
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

def create_resnet(weight_path=None, depth=50):
    cfg = {
            50: ([3, 4, 6, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck),
            152: ([3, 8, 36, 3], bottleneck),
            }
    net = ffnet.FeedForwardNet()
    net.add(Conv2D('input-conv', 64, 7, 2, pad=3, input_sample_shape=(3, 224, 224)))
    net.add(BatchNormalization('input-bn'))
    net.add(Activation('input_relu'))
    net.add(MaxPooling2D('input_pool', 3, 2, pad=1))

    conf = cfg[depth]
    stage(0, net, conf[0][0], 64, 64, 256, 1, conf[1])
    stage(1, net, conf[0][1], 256, 128, 512, 2, conf[1])
    stage(2, net, conf[0][2], 512, 256, 1024, 2, conf[1])
    stage(3, net, conf[0][3], 1024, 512, 2048, 2, conf[1])
    net.add(AvgPooling2D('avg', 7, 1))
    net.add(Flatten('flat'))
    net.add(Dense('dense', 1000))

    init_params(net, weight_path)
    return net


def create_wide_resnet(weight_path=None):
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

    init_params(net, weight_path)
    return net


if __name__ == '__main__':
    create_net('wrn-50-2.pickle')
