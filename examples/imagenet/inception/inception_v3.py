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

Refer to
https://github.com/tensorflow/models/blob/master/slim/nets/inception_v3.py
"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from singa.layer import Conv2D, Activation, MaxPooling2D, AvgPooling2D,\
        Split, Concat, Dropout, Flatten, BatchNormalization

from singa import net as ffnet

ffnet.verbose = True


def conv2d(net, name, nb_filter, k, s=1, border_mode='SAME', src=None):
    if type(k) is list:
        k = (k[0], k[1])
    net.add(Conv2D(name, nb_filter, k, s, border_mode=border_mode,
                   use_bias=False), src)
    net.add(BatchNormalization('%s/BatchNorm' % name))
    return net.add(Activation(name+'/relu'))


def inception_v3_base(name, sample_shape, final_endpoint, aux_endpoint,
                      depth_multiplier=1, min_depth=16):
    """Creates the Inception V3 network up to the given final endpoint.

    Args:
        sample_shape: input image sample shape, 3d tuple
        final_endpoint: specifies the endpoint to construct the network up to.
        aux_endpoint: for aux loss.

    Returns:
        logits: the logits outputs of the model.
        end_points: the set of end_points from the inception model.

    Raises:
        ValueError: if final_endpoint is not set to one of the predefined values
    """
    V3 = 'InceptionV3'
    end_points = {}
    net = ffnet.FeedForwardNet()

    def final_aux_check(block_name):
        if block_name == final_endpoint:
            return True
        if block_name == aux_endpoint:
            aux = aux_endpoint + '-aux'
            end_points[aux] = net.add(Split(aux, 2))
        return False

    def depth(d):
        return max(int(d * depth_multiplier), min_depth)

    blk = V3 + '/Conv2d_1a_3x3'
    # 299 x 299 x 3
    net.add(Conv2D(blk, depth(32), 3, 2, border_mode='VALID', use_bias=False,
                   input_sample_shape=sample_shape))
    net.add(BatchNormalization(blk + '/BatchNorm'))
    end_points[blk] = net.add(Activation(blk + '/relu'))
    if final_aux_check(blk):
        return net, end_points

    # 149 x 149 x 32
    conv2d(net, '%s/Conv2d_2a_3x3' % V3, depth(32), 3, border_mode='VALID')
    # 147 x 147 x 32
    conv2d(net, '%s/Conv2d_2b_3x3' % V3, depth(64), 3)
    # 147 x 147 x 64
    net.add(MaxPooling2D('%s/MaxPool_3a_3x3' % V3, 3, 2, border_mode='VALID'))
    # 73 x 73 x 64
    conv2d(net, '%s/Conv2d_3b_1x1' % V3, depth(80), 1, border_mode='VALID')
    # 73 x 73 x 80.
    conv2d(net, '%s/Conv2d_4a_3x3' % V3, depth(192), 3, border_mode='VALID')
    # 71 x 71 x 192.
    net.add(MaxPooling2D('%s/MaxPool_5a_3x3' % V3, 3, 2, border_mode='VALID'))

    # 35 x 35 x 192.
    blk = V3 + '/Mixed_5b'
    s = net.add(Split('%s/Split' % blk, 4))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_0a_1x1' % blk, depth(64), 1, src=s)
    conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % blk, depth(48), 1, src=s)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0b_5x5' % blk, depth(64), 5)
    conv2d(net, '%s/Branch_2/Conv2d_0a_1x1' % blk, depth(64), 1, src=s)
    conv2d(net, '%s/Branch_2/Conv2d_0b_3x3' % blk, depth(96), 3)
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0c_3x3' % blk, depth(96), 3)
    net.add(AvgPooling2D('%s/Branch_3/AvgPool_0a_3x3' % blk, 3, 1), s)
    br3 = conv2d(net, '%s/Branch_3/Conv2d_0b_1x1' % blk, depth(32), 1)
    end_points[blk] = net.add(Concat('%s/Concat' % blk, 1),
                              [br0, br1, br2, br3])
    if final_aux_check(blk):
        return net, end_points

    # mixed_1: 35 x 35 x 288.
    blk = V3 + '/Mixed_5c'
    s = net.add(Split('%s/Split' % blk, 4))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_0a_1x1' % blk, depth(64), 1, src=s)
    conv2d(net, '%s/Branch_1/Conv2d_0b_1x1' % blk, depth(48), 1, src=s)
    br1 = conv2d(net, '%s/Branch_1/Conv_1_0c_5x5' % blk, depth(64), 5)
    conv2d(net, '%s/Branch_2/Conv2d_0a_1x1' % blk, depth(64), 1, src=s)
    conv2d(net, '%s/Branch_2/Conv2d_0b_3x3' % blk, depth(96), 3)
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0c_3x3' % blk, depth(96), 3)
    net.add(AvgPooling2D('%s/Branch_3/AvgPool_0a_3x3' % blk, 3, 1), src=s)
    br3 = conv2d(net, '%s/Branch_3/Conv2d_0b_1x1' % blk, depth(64), 1)
    end_points[blk] = net.add(Concat('%s/Concat' % blk, 1),
                              [br0, br1, br2, br3])
    if final_aux_check(blk):
        return net, end_points

    # mixed_2: 35 x 35 x 288.
    blk = V3 + '/Mixed_5d'
    s = net.add(Split('%s/Split' % blk, 4))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_0a_1x1' % blk, depth(64), 1, src=s)
    conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % blk, depth(48), 1, src=s)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0b_5x5' % blk, depth(64), 5)
    conv2d(net, '%s/Branch_2/Conv2d_0a_1x1' % blk, depth(64), 1, src=s)
    conv2d(net, '%s/Branch_2/Conv2d_0b_3x3' % blk, depth(96), 3)
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0c_3x3' % blk, depth(96), 3)
    net.add(AvgPooling2D('%s/Branch_3/AvgPool_0a_3x3' % blk,  3, 1), s)
    br3 = conv2d(net, '%s/Branch_3/Conv2d_0b_1x1' % blk, depth(64), 1)
    end_points[blk] = net.add(Concat('%s/Concat' % blk, 1),
                              [br0, br1, br2, br3])
    if final_aux_check(blk):
        return net, end_points

    # mixed_3: 17 x 17 x 768.
    blk = V3 + '/Mixed_6a'
    s = net.add(Split('%s/Split' % blk, 3))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_1a_1x1' % blk, depth(384), 3, 2,
                 border_mode='VALID', src=s)
    conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % blk, depth(64), 1, src=s)
    conv2d(net, '%s/Branch_1/Conv2d_0b_3x3' % blk, depth(96), 3)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_1a_1x1' % blk, depth(96), 3, 2,
                 border_mode='VALID')
    br2 = net.add(MaxPooling2D('%s/Branch_2/MaxPool_1a_3x3' % blk, 3, 2,
                               border_mode='VALID'), s)
    end_points[blk] = net.add(Concat('%s/Concat' % blk, 1),  [br0, br1, br2])
    if final_aux_check(blk):
        return net, end_points

    # mixed4: 17 x 17 x 768.
    blk = V3 + '/Mixed_6b'
    s = net.add(Split('%s/Split' % blk, 4))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_0a_1x1' % blk, depth(192), 1, src=s)
    conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % blk, depth(128), 1, src=s)
    conv2d(net, '%s/Branch_1/Conv2d_0b_1x7' % blk, depth(128), [1, 7])
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0c_7x1' % blk, depth(192), [7, 1])
    conv2d(net, '%s/Branch_2/Conv2d_0a_1x1' % blk, depth(128), [1, 1], src=s)
    conv2d(net, '%s/Branch_2/Conv2d_0b_7x1' % blk, depth(128), [7, 1])
    conv2d(net, '%s/Branch_2/Conv2d_0c_1x7' % blk, depth(128), [1, 7])
    conv2d(net, '%s/Branch_2/Conv2d_0d_7x1' % blk, depth(128), [7, 1])
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0e_1x7' % blk, depth(192), [1, 7])
    net.add(AvgPooling2D('%s/Branch_3/AvgPool_0a_3x3' % blk, 3, 1), s)
    br3 = conv2d(net, '%s/Branch_3/Conv2d_0b_1x1' % blk, depth(192), [1, 1])
    end_points[blk] = net.add(Concat('%s/Concat' % blk, 1),
                              [br0, br1, br2, br3])
    if final_aux_check(blk):
        return net, end_points

    # mixed_5: 17 x 17 x 768.
    blk = V3 + '/Mixed_6c'
    s = net.add(Split('%s/Split' % blk, 4))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_0a_1x1' % blk, depth(192), [1, 1],
                 src=s)
    conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % blk, depth(160), [1, 1], src=s)
    conv2d(net, '%s/Branch_1/Conv2d_0b_1x7' % blk, depth(160), [1, 7])
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0c_7x1' % blk, depth(192), [7, 1])
    conv2d(net, '%s/Branch_2/Conv2d_0a_1x1' % blk, depth(160), [1, 1], src=s)
    conv2d(net, '%s/Branch_2/Conv2d_0b_7x1' % blk, depth(160), [7, 1])
    conv2d(net, '%s/Branch_2/Conv2d_0c_1x7' % blk, depth(160), [1, 7])
    conv2d(net, '%s/Branch_2/Conv2d_0d_7x1' % blk, depth(160), [7, 1])
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0e_1x7' % blk, depth(192), [1, 7])
    net.add(AvgPooling2D('%s/Branch_3/AvgPool_0a_3x3' % blk, 3, 1), s)
    br3 = conv2d(net, '%s/Branch_3/Conv2d_0b_1x1' % blk, depth(192), [1, 1])
    end_points[blk] = net.add(Concat('%s/Concat' % blk, 1),
                              [br0, br1, br2, br3])
    if final_aux_check(blk):
        return net, end_points

    # mixed_6: 17 x 17 x 768.
    blk = V3 + '/Mixed_6d'
    s = net.add(Split('%s/Split' % blk, 4))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_0a_1x1' % blk, depth(192), [1, 1],
                 src=s)
    conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % blk, depth(160), [1, 1], src=s)
    conv2d(net, '%s/Branch_1/Conv2d_0b_1x7' % blk, depth(160), [1, 7])
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0c_7x1' % blk, depth(192), [7, 1])
    conv2d(net, '%s/Branch_2/Conv2d_0a_1x1' % blk, depth(160), [1, 1], src=s)
    conv2d(net, '%s/Branch_2/Conv2d_0b_7x1' % blk, depth(160), [7, 1])
    conv2d(net, '%s/Branch_2/Conv2d_0c_1x7' % blk, depth(160), [1, 7])
    conv2d(net, '%s/Branch_2/Conv2d_0d_7x1' % blk, depth(160), [7, 1])
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0e_1x7' % blk, depth(192), [1, 7])
    net.add(AvgPooling2D('%s/Branch_3/AvgPool_0a_3x3' % blk, 3, 1), s)
    br3 = conv2d(net, '%s/Branch_3/Conv2d_0b_1x1' % blk, depth(192), [1, 1])
    end_points[blk] = net.add(Concat('%s/Concat' % blk, 1),
                              [br0, br1, br2, br3])
    if final_aux_check(blk):
        return net, end_points

    blk = V3 + '/Mixed_6e'
    s = net.add(Split('%s/Split' % blk, 4))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_0a_1x1' % blk, depth(192), [1, 1],
                 src=s)
    conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % blk, depth(192), [1, 1], src=s)
    conv2d(net, '%s/Branch_1/Conv2d_0b_1x7' % blk, depth(192), [1, 7])
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0c_7x1' % blk, depth(192), [7, 1])
    conv2d(net, '%s/Branch_2/Conv2d_0a_1x1' % blk, depth(192), [1, 1], src=s)
    conv2d(net, '%s/Branch_2/Conv2d_0b_7x1' % blk, depth(192), [7, 1])
    conv2d(net, '%s/Branch_2/Conv2d_0c_1x7' % blk, depth(192), [1, 7])
    conv2d(net, '%s/Branch_2/Conv2d_0d_7x1' % blk, depth(192), [7, 1])
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0e_1x7' % blk, depth(192), [1, 7])
    net.add(AvgPooling2D('%s/Branch_3/AvgPool_0a_3x3' % blk, 3, 1), s)
    br3 = conv2d(net, '%s/Branch_3/Conv2d_0b_1x1' % blk, depth(192), [1, 1])
    end_points[blk] = net.add(Concat('%s/Concat' % blk, 1),
                              [br0, br1, br2, br3])
    if final_aux_check(blk):
        return net, end_points

    # mixed_8: 8 x 8 x 1280.
    blk = V3 + '/Mixed_7a'
    s = net.add(Split('%s/Split' % blk, 3))
    conv2d(net, '%s/Branch_0/Conv2d_0a_1x1' % blk, depth(192), [1, 1], src=s)
    br0 = conv2d(net, '%s/Branch_0/Conv2d_1a_3x3' % blk, depth(320), [3, 3], 2,
                 border_mode='VALID')
    conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % blk, depth(192), [1, 1], src=s)
    conv2d(net, '%s/Branch_1/Conv2d_0b_1x7' % blk, depth(192), [1, 7])
    conv2d(net, '%s/Branch_1/Conv2d_0c_7x1' % blk, depth(192), [7, 1])
    br1 = conv2d(net, '%s/Branch_1/Conv2d_1a_3x3' % blk, depth(192), [3, 3], 2,
                 border_mode='VALID')
    br2 = net.add(MaxPooling2D('%s/Branch_2/MaxPool_1a_3x3' % blk, 3, 2,
                               border_mode='VALID'), s)
    end_points[blk] = net.add(Concat('%s/Concat' % blk, 1),  [br0, br1, br2])
    if final_aux_check(blk):
        return net, end_points

    # mixed_9: 8 x 8 x 2048.
    blk = V3 + '/Mixed_7b'
    s = net.add(Split('%s/Split' % blk, 4))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_0a_1x1' % blk, depth(320), 1, src=s)
    conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % blk, depth(384), 1, src=s)
    s1 = net.add(Split('%s/Branch_1/Split1' % blk, 2))
    br11 = conv2d(net, '%s/Branch_1/Conv2d_0b_1x3' % blk, depth(384), [1, 3],
                  src=s1)
    br12 = conv2d(net, '%s/Branch_1/Conv2d_0b_3x1' % blk, depth(384), [3, 1],
                  src=s1)
    br1 = net.add(Concat('%s/Branch_1/Concat1' % blk, 1),  [br11, br12])
    conv2d(net, '%s/Branch_2/Conv2d_0a_1x1' % blk, depth(448), 1, src=s)
    conv2d(net, '%s/Branch_2/Conv2d_0b_3x3' % blk, depth(384), 3)
    s2 = net.add(Split('%s/Branch_2/Split2' % blk, 2))
    br21 = conv2d(net, '%s/Branch_2/Conv2d_0c_1x3' % blk, depth(384), [1, 3],
                  src=s2)
    br22 = conv2d(net, '%s/Branch_2/Conv2d_0d_3x1' % blk, depth(384), [3, 1],
                  src=s2)
    br2 = net.add(Concat('%s/Branch_2/Concat2' % blk, 1),  [br21, br22])
    net.add(AvgPooling2D('%s/Branch_3/AvgPool_0a_3x3' % blk, 3, 1), src=s)
    br3 = conv2d(net, '%s/Branch_3/Conv2d_0b_1x1' % blk, depth(192), [1, 1])
    end_points[blk] = net.add(Concat('%s/Concat' % blk, 1),
                              [br0, br1, br2, br3])
    if final_aux_check(blk):
        return net, end_points

    # mixed_10: 8 x 8 x 2048.
    blk = V3 + '/Mixed_7c'
    s = net.add(Split('%s/Split' % blk, 4))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_0a_1x1' % blk, depth(320), 1, src=s)
    conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % blk, depth(384), 1, src=s)
    s1 = net.add(Split('%s/Branch_1/Split1' % blk, 2))
    br11 = conv2d(net, '%s/Branch_1/Conv2d_0b_1x3' % blk, depth(384), [1, 3],
                  src=s1)
    br12 = conv2d(net, '%s/Branch_1/Conv2d_0c_3x1' % blk, depth(384), [3, 1],
                  src=s1)
    br1 = net.add(Concat('%s/Branch_1/Concat1' % blk, 1),  [br11, br12])
    conv2d(net, '%s/Branch_2/Conv2d_0a_1x1' % blk, depth(448), [1, 1],
                 src=s)
    conv2d(net, '%s/Branch_2/Conv2d_0b_3x3' % blk, depth(384), [3, 3])
    s2 = net.add(Split('%s/Branch_2/Split2' % blk, 2))
    br21 = conv2d(net, '%s/Branch_2/Conv2d_0c_1x3' % blk, depth(384), [1, 3],
                  src=s2)
    br22 = conv2d(net, '%s/Branch_2/Conv2d_0d_3x1' % blk, depth(384), [3, 1],
                  src=s2)
    br2 = net.add(Concat('%s/Branch_2/Concat2' % blk, 1),  [br21, br22])
    net.add(AvgPooling2D('%s/Branch_3/AvgPool_0a_3x3' % blk, 3, 1), src=s)
    br3 = conv2d(net, '%s/Branch_3/Conv2d_0b_1x1' % blk, depth(192), [1, 1])
    end_points[blk] = net.add(Concat('%s/Concat' % blk, 1),
                              [br0, br1, br2, br3])
    assert final_endpoint == blk, \
        'final_enpoint = %s is not in the net' % final_endpoint
    return net, end_points


def create_net(num_classes=1001, sample_shape=(3, 299, 299), is_training=True,
               final_endpoint='InceptionV3/Mixed_7c',
               aux_endpoint='InceptionV3/Mixed_6e',
               dropout_keep_prob=0.8):
    """Creates the Inception V4 model.

    Args:
        num_classes: number of predicted classes.
        is_training: whether is training or not.
        dropout_keep_prob: float, the fraction to keep before final layer.
        final_endpoint: 'InceptionV3/Mixed_7d',
        aux_endpoint:

    Returns:
        logits: the logits outputs of the model.
        end_points: the set of end_points from the inception model.
    """
    name = 'InceptionV3'
    net, end_points = inception_v3_base(name, sample_shape, final_endpoint,
                                        aux_endpoint)
    # Auxiliary Head logits
    if aux_endpoint is not None:
        # 8 x 8 x 1280
        aux_logits = end_points[aux_endpoint + '-aux']
        blk = name + '/AuxLogits'
        net.add(AvgPooling2D('%s/AvgPool_1a_5x5' % blk, 5, stride=3,
                             border_mode='VALID'), aux_logits)
        t = conv2d(net, '%s/Conv2d_1b_1x1' % blk, 128, 1)
        s = t.get_output_sample_shape()[1:3]
        conv2d(net, '%s/Conv2d_2a_%dx%d' % (blk, s[0], s[1]), 768, s,
               border_mode='VALID')
        net.add(Conv2D('%s/Conv2d_2b_1x1' % blk, num_classes, 1))
        net.add(Flatten('%s/flat' % blk))

    # Final pooling and prediction
    # 8 x 8 x 2048
    blk = name + '/Logits'
    last_layer = end_points[final_endpoint]
    net.add(AvgPooling2D('%s/AvgPool_1a' % blk,
                         last_layer.get_output_sample_shape()[1:3], 1,
                         border_mode='VALID'), last_layer)
    # 1 x 1 x 2048
    net.add(Dropout('%s/Dropout_1b' % blk, 1 - dropout_keep_prob))
    net.add(Conv2D('%s/Conv2d_1c_1x1' % blk, num_classes, 1))
    end_points[blk] = net.add(Flatten('%s/flat' % blk))
    # 2048
    return net, end_points


if __name__ == '__main__':
    net, _ = create_net()
