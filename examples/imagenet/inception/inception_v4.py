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
http://arxiv.org/abs/1602.07261.

  Inception-v4, Inception-ResNet and the Impact of Residual Connections
    on Learning
  Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi

Refer to
https://github.com/tensorflow/models/blob/master/slim/nets/inception_v4.py
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from singa.layer import Conv2D, Activation, MaxPooling2D, AvgPooling2D,\
        Split, Concat, Dropout, Flatten, Dense, BatchNormalization

from singa import net as ffnet

ffnet.verbose = True


def conv2d(net, name, nb_filter, k, s=1, border_mode='SAME', src=None):
    net.add(Conv2D(name, nb_filter, k, s, border_mode=border_mode,
                   use_bias=False), src)
    net.add(BatchNormalization('%s/BatchNorm' % name))
    return net.add(Activation(name+'/relu'))


def block_inception_a(blk, net):
    """Builds Inception-A block for Inception v4 network."""
    # By default use stride=1 and SAME padding
    s = net.add(Split('%s/Split' % blk, 4))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_0a_1x1' % blk, 96, 1, src=s)
    conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % blk, 64, 1, src=s)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0b_3x3' % blk, 96, 3)
    conv2d(net, '%s/Branch_2/Conv2d_0a_1x1' % blk, 64, 1, src=s)
    conv2d(net, '%s/Branch_2/Conv2d_0b_3x3' % blk, 96, 3)
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0c_3x3' % blk, 96, 3)
    net.add(AvgPooling2D('%s/Branch_3/AvgPool_0a_3x3' % blk, 3, stride=1), s)
    br3 = conv2d(net, '%s/Branch_3/Conv2d_0b_1x1' % blk, 96, 1)
    return net.add(Concat('%s/Concat' % blk, 1), [br0, br1, br2, br3])


def block_reduction_a(blk, net):
    """Builds Reduction-A block for Inception v4 network."""
    # By default use stride=1 and SAME padding
    s = net.add(Split('%s/Split' % blk, 3))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_1a_3x3' % blk, 384, 3, 2,
                 border_mode='VALID', src=s)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % blk, 192, 1, src=s)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0b_3x3' % blk, 224, 3)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_1a_3x3' % blk, 256, 3, 2,
                 border_mode='VALID')
    br2 = net.add(MaxPooling2D('%s/Branch_2/MaxPool_1a_3x3' % blk, 3, 2,
                               border_mode='VALID'), s)
    return net.add(Concat('%s/Concat' % blk, 1), [br0, br1, br2])


def block_inception_b(blk, net):
    """Builds Inception-B block for Inception v4 network."""
    # By default use stride=1 and SAME padding
    s = net.add(Split('%s/Split' % blk, 4))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_0a_1x1' % blk, 384, 1, src=s)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % blk, 192, 1, src=s)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0b_1x7' % blk, 224, (1, 7))
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0c_7x1' % blk, 256, (7, 1))
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0a_1x1' % blk, 192, 1, src=s)
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0b_7x1' % blk, 192, (7, 1))
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0c_1x7' % blk, 224, (1, 7))
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0d_7x1' % blk, 224, (7, 1))
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0e_1x7' % blk, 256, (1, 7))
    br3 = net.add(AvgPooling2D('%s/Branch_3/AvgPool_0a_3x3' % blk, 3, 1), s)
    br3 = conv2d(net, '%s/Branch_3/Conv2d_0b_1x1' % blk, 128, 1)
    return net.add(Concat('%s/Concat' % blk, 1), [br0, br1, br2, br3])


def block_reduction_b(blk, net):
    """Builds Reduction-B block for Inception v4 network."""
    # By default use stride=1 and SAME padding
    s = net.add(Split('%s/Split' % blk, 3))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_0a_1x1' % blk, 192, 1, src=s)
    br0 = conv2d(net, '%s/Branch_0/Conv2d_1a_3x3' % blk, 192, 3, 2,
                 border_mode='VALID')
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % blk, 256, 1, src=s)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0b_1x7' % blk, 256, (1, 7))
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0c_7x1' % blk, 320, (7, 1))
    br1 = conv2d(net, '%s/Branch_1/Conv2d_1a_3x3' % blk, 320, 3, 2,
                 border_mode='VALID')
    br2 = net.add(MaxPooling2D('%s/Branch_2/MaxPool_1a_3x3' % blk, 3, 2,
                               border_mode='VALID'), s)
    return net.add(Concat('%s/Concat' % blk, 1), [br0, br1, br2])


def block_inception_c(blk, net):
    """Builds Inception-C block for Inception v4 network."""
    # By default use stride=1 and SAME padding
    s = net.add(Split('%s/Split' % blk, 4))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_0a_1x1' % blk, 256, 1, src=s)

    br1 = conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % blk, 384, 1, src=s)
    br1 = net.add(Split('%s/Branch_1/Split' % blk, 2))
    br10 = conv2d(net, '%s/Branch_1/Conv2d_0b_1x3' % blk, 256, (1, 3), src=br1)
    br11 = conv2d(net, '%s/Branch_1/Conv2d_0c_3x1' % blk, 256, (3, 1), src=br1)
    br1 = net.add(Concat('%s/Branch_1/Concat' % blk, 1), [br10, br11])

    br2 = conv2d(net, '%s/Branch_2/Conv2d_0a_1x1' % blk, 384, 1, src=s)
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0b_3x1' % blk, 448, (3, 1))
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0c_1x3' % blk, 512, (1, 3))
    br2 = net.add(Split('%s/Branch_2/Split' % blk, 2))
    br20 = conv2d(net, '%s/Branch_2/Conv2d_0d_1x3' % blk, 256, (1, 3), src=br2)
    br21 = conv2d(net, '%s/Branch_2/Conv2d_0e_3x1' % blk, 256, (3, 1), src=br2)
    br2 = net.add(Concat('%s/Branch_2/Concat' % blk, 1), [br20, br21])

    br3 = net.add(AvgPooling2D('%s/Branch_3/AvgPool_0a_3x3' % blk, 3, 1), s)
    br3 = conv2d(net, '%s/Branch_3/Conv2d_0b_1x1' % blk, 256, 1)
    return net.add(Concat('%s/Concat' % blk, 1), [br0, br1, br2, br3])


def inception_v4_base(sample_shape, final_endpoint='Inception/Mixed_7d',
                      aux_endpoint='Inception/Mixed_6e'):
    """Creates the Inception V4 network up to the given final endpoint.

    Endpoint name list: 'InceptionV4/' +
        ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
        'Mixed_3a', 'Mixed_4a', 'Mixed_5a', 'Mixed_5b', 'Mixed_5c', 'Mixed_5d',
        'Mixed_5e', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d', 'Mixed_6e',
        'Mixed_6f', 'Mixed_6g', 'Mixed_6h', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c',
        'Mixed_7d']

    Args:
        sample_shape: input image sample shape, 3d tuple
        final_endpoint: specifies the endpoint to construct the network up to.
        aux_endpoint: for aux loss.

    Returns:
        the neural net
        the set of end_points from the inception model.
    """
    name = 'InceptionV4'
    end_points = {}
    net = ffnet.FeedForwardNet()

    def final_aux_check(block_name):
        if block_name == final_endpoint:
            return True
        if block_name == aux_endpoint:
            aux = aux_endpoint + '-aux'
            end_points[aux] = net.add(Split(aux, 2))
        return False

    # 299 x 299 x 3
    blk = name + '/Conv2d_1a_3x3'
    net.add(Conv2D(blk, 32, 3, 2, border_mode='VALID', use_bias=False,
                   input_sample_shape=sample_shape))
    net.add(BatchNormalization('%s/BatchNorm' % blk))
    end_points[blk] = net.add(Activation('%s/relu' % blk))
    if final_aux_check(blk):
        return net, end_points

    # 149 x 149 x 32
    blk = name + '/Conv2d_2a_3x3'
    end_points[blk] = conv2d(net, blk, 32, 3, border_mode='VALID')
    if final_aux_check(blk):
        return net, end_points

    # 147 x 147 x 32
    blk = name + '/Conv2d_2b_3x3'
    end_points[blk] = conv2d(net, blk, 64, 3)
    if final_aux_check(blk):
        return net, end_points

    # 147 x 147 x 64
    blk = name + '/Mixed_3a'
    s = net.add(Split('%s/Split' % blk, 2))
    br0 = net.add(MaxPooling2D('%s/Branch_0/MaxPool_0a_3x3' % blk, 3, 2,
                               border_mode='VALID'), s)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0a_3x3' % blk, 96, 3, 2,
                 border_mode='VALID', src=s)
    end_points[blk] = net.add(Concat('%s/Concat' % blk, 1), [br0, br1])
    if final_aux_check(blk):
        return net, end_points

    # 73 x 73 x 160
    blk = name + '/Mixed_4a'
    s = net.add(Split('%s/Split' % blk, 2))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_0a_1x1' % blk, 64, 1, src=s)
    br0 = conv2d(net, '%s/Branch_0/Conv2d_1a_3x3' % blk, 96, 3,
                 border_mode='VALID')
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % blk, 64, 1, src=s)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0b_1x7' % blk, 64, (1, 7))
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0c_7x1' % blk, 64, (7, 1))
    br1 = conv2d(net, '%s/Branch_1/Conv2d_1a_3x3' % blk, 96, 3,
                 border_mode='VALID')
    end_points[blk] = net.add(Concat('%s/Concat' % blk, 1), [br0, br1])
    if final_aux_check(blk):
        return net, end_points

    # 71 x 71 x 192
    blk = name + '/Mixed_5a'
    s = net.add(Split('%s/Split' % blk, 2))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_1a_3x3' % blk, 192, 3, 2,
                 border_mode='VALID', src=s)
    br1 = net.add(MaxPooling2D('%s/Branch_1/MaxPool_1a_3x3' % blk, 3, 2,
                               border_mode='VALID'), s)
    end_points[blk] = net.add(Concat('%s/Concat' % blk, 1), [br0, br1])
    if final_aux_check(blk):
        return net, end_points

    # 35 x 35 x 384
    # 4 x Inception-A blocks
    for idx in range(4):
        blk = name + '/Mixed_5' + chr(ord('b') + idx)
        end_points[blk] = block_inception_a(blk, net)
        if final_aux_check(blk):
            return net, end_points

    # 35 x 35 x 384
    # Reduction-A block
    blk = name + '/Mixed_6a'
    end_points[blk] = block_reduction_a(blk, net)
    if final_aux_check(blk):
        return net, end_points[blk], end_points

    # 17 x 17 x 1024
    # 7 x Inception-B blocks
    for idx in range(7):
        blk = name + '/Mixed_6' + chr(ord('b') + idx)
        end_points[blk] = block_inception_b(blk, net)
        if final_aux_check(blk):
            return net, end_points

    # 17 x 17 x 1024
    # Reduction-B block
    blk = name + '/Mixed_7a'
    end_points[blk] = block_reduction_b(blk, net)
    if final_aux_check(blk):
        return net, end_points

    # 8 x 8 x 1536
    # 3 x Inception-C blocks
    for idx in range(3):
        blk = name + '/Mixed_7' + chr(ord('b') + idx)
        end_points[blk] = block_inception_c(blk, net)
        if final_aux_check(blk):
            return net, end_points

    assert final_endpoint == blk, \
        'final_enpoint = %s is not in the net' % final_endpoint


def create_net(num_classes=1001, sample_shape=(3, 299, 299), is_training=True,
               dropout_keep_prob=0.8, final_endpoint='InceptionV4/Mixed_7d',
               aux_endpoint='InceptionV4/Mixed_6e'):
    """Creates the Inception V4 model.

    Args:
        num_classes: number of predicted classes.
        is_training: whether is training or not.
        dropout_keep_prob: float, the fraction to keep before final layer.
        final_endpoint, aux_endpoint: refer to inception_v4_base()

    Returns:
        logits: the logits outputs of the model.
        end_points: the set of end_points from the inception model.
    """
    end_points = {}
    name = 'InceptionV4'
    net, end_points = inception_v4_base(sample_shape,
                                        final_endpoint=final_endpoint,
                                        aux_endpoint=aux_endpoint)
    # Auxiliary Head logits
    if aux_endpoint is not None:
        # 17 x 17 x 1024
        aux_logits = end_points[aux_endpoint + '-aux']
        blk = name + '/AuxLogits'
        net.add(AvgPooling2D('%s/AvgPool_1a_5x5' % blk, 5, stride=3,
                             border_mode='VALID'), aux_logits)
        t = conv2d(net, '%s/Conv2d_1b_1x1' % blk, 128, 1)
        conv2d(net, '%s/Conv2d_2a' % blk, 768,
               t.get_output_sample_shape()[1:3], border_mode='VALID')
        net.add(Flatten('%s/flat' % blk))
        end_points[blk] = net.add(Dense('%s/Aux_logits' % blk, num_classes))

    # Final pooling and prediction
    # 8 x 8 x 1536
    blk = name + '/Logits'
    last_layer = end_points[final_endpoint]
    net.add(AvgPooling2D('%s/AvgPool_1a' % blk,
                         last_layer.get_output_sample_shape()[1:3],
                         border_mode='VALID'),
            last_layer)
    # 1 x 1 x 1536
    net.add(Dropout('%s/Dropout_1b' % blk, 1 - dropout_keep_prob))
    net.add(Flatten('%s/PreLogitsFlatten' % blk))
    # 1536
    end_points[blk] = net.add(Dense('%s/Logits' % blk, num_classes))
    return net, end_points


if __name__ == '__main__':
    net, _ = create_net()
