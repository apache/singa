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
        Split, Concat, Dropout, Flatten, Dense, BatchNormalization

from singa import net as ffnet

ffnet.verbose = True

def conv2d(net, name, nb_filter, k, s=1, border_mode='SAME', src=None):
    if type(k) is list:
        k = (k[0], k[1])
    net.add(Conv2D(name, nb_filter, k, s, border_mode=border_mode, use_bias=False), src)
    net.add(BatchNormalization('%s/BatchNorm' % name))
    return net.add(Activation(name+'/relu'))


def inception_v3_base(name, sample_shape, final_endpoint='Mixed_6e', aux_name=None, depth_multiplier=1, min_depth=16):
    """Creates the Inception V3 network up to the given final endpoint.

    Args:
        inputs: a 4-D tensor of size [batch_size, height, width, 3].
        final_endpoint: specifies the endpoint to construct the network up to.

    Returns:
        logits: the logits outputs of the model.
        end_points: the set of end_points from the inception model.

    Raises:
        ValueError: if final_endpoint is not set to one of the predefined values,
    """
    endpoints = {}
    def final_aux_check(block_name, net):
        if block_name == final_endpoint:
            return net, endpoints[block_name], endpoints
        if block_name == aux_name:
            endpoints[aux_name + '-aux'] = net.add(Split('%s-aux' % aux_name, 2))

    net = ffnet.FeedForwardNet()
    depth = lambda d: max(int(d * depth_multiplier), min_depth)
    V3 = 'InceptionV3'

    name = V3 + '/Conv2d_1a_3x3'
    # 299 x 299 x 3
    net.add(Conv2D(name, depth(32), 3, 2, border_mode='VALID', use_bias=False, input_sample_shape=sample_shape))
    net.add(BatchNormalization(name + '/BatchNorm'))
    net.add(Activation(name + '/relu'))
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


    m5b = V3 + '/Mixed_5b'
    s = net.add(Split('%s/Split' % m5b, 4))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_0a_1x1' % m5b, depth(64), 1, src=s)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % m5b, depth(48), 1, src=s)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0b_5x5' % m5b, depth(64), 5)
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0a_1x1' % m5b, depth(64), 1, src=s)
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0b_3x3' % m5b, depth(96), 3)
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0c_3x3' % m5b, depth(96), 3)
    net.add(AvgPooling2D('%s/Branch_3/AvgPool_0a_3x3' % m5b, 3, 1), s)
    br3 = conv2d(net, '%s/Branch_3/Conv2d_0b_1x1' % m5b, depth(32), 1)
    endpoints[m5b] =net.add(Concat('%s/Concat' % m5b, 1),  [br0, br1, br2, br3])
    final_aux_check(m5b, net)
    # mixed_1: 35 x 35 x 288.
    m5c = V3 + '/Mixed_5c'
    s = net.add(Split('%s/Split' % m5c, 4))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_0a_1x1' % m5c, depth(64), 1, src=s)
    br1 = conv2d(net, '%s/Branch_1/Con2d_0b_1x1' % m5c, depth(48), 1, src=s)
    br1 = conv2d(net, '%s/Branch_1/Conv_1_0c_5x5' % m5c, depth(64), 5)
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0a_1x1' % m5c, depth(64), 1, src=s)
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0b_3x3' % m5c, depth(96), 3)
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0c_3x3' % m5c, depth(96), 3)
    br3 = net.add(AvgPooling2D('%s/Branch_3/AvgPool_0a_3x3' % m5c, 3, 1), src=s)
    br3 = conv2d(net, '%s/Branch_3/Conv2d_0b_1x1' % m5c, depth(64), 1)
    endpoints[m5c] = net.add(Concat('%s/Concat' % m5c, 1),  [br0, br1, br2, br3])
    final_aux_check(m5c, net)

    # mixed_2: 35 x 35 x 288.
    m5d = V3 + '/Mixed_5d'
    s = net.add(Split('%s/Split' % m5d, 4))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_0a_1x1' % m5d, depth(64), 1, src=s)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % m5d, depth(48), 1, src=s)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0b_5x5' % m5d, depth(64), 5)
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0a_1x1' % m5d, depth(64), 1, src=s)
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0b_3x3' % m5d, depth(96), 3)
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0c_3x3' % m5d, depth(96), 3)
    br3 = net.add(AvgPooling2D('%s/Branch_3/AvgPool_0a_3x3' % m5d,  3, 1), s)
    br3 = conv2d(net, '%s/Branch_3/Conv2d_0b_1x1' % m5d, depth(64), 1)
    endpoints[m5d] =net.add(Concat('%s/Concat' % m5d, 1),  [br0, br1, br2, br3])
    final_aux_check(m5d, net)

    # mixed_3: 17 x 17 x 768.
    m6a = V3 + '/Mixed_6a'
    s = net.add(Split('%s/Split' % m6a, 3))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_1a_1x1' % m6a, depth(384), 3, 2, border_mode='VALID', src=s)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % m6a, depth(64), 1, src=s)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0b_3x3' % m6a, depth(96), 3)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_1a_1x1' % m6a, depth(96), 3, 2, border_mode='VALID')
    br2 = net.add(MaxPooling2D('%s/Branch_2/MaxPool_1a_3x3' % m6a, 3, 2, border_mode='VALID'), s)
    endpoints[m6a] = net.add(Concat('%s/Concat' % m6a, 1),  [br0, br1, br2])
    final_aux_check(m6a, net)

    # mixed4: 17 x 17 x 768.
    m6b = V3 + '/Mixed_6b'
    s = net.add(Split('%s/Split' % m6b, 4))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_0a_1x1' % m6b, depth(192), 1, src=s)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % m6b, depth(128), 1, src=s)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0b_1x7' % m6b, depth(128), [1, 7])
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0c_7x1' % m6b, depth(192), [7, 1])
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0a_1x1' % m6b, depth(128), [1, 1], src=s)
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0b_7x1' % m6b, depth(128), [7, 1])
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0c_1x7' % m6b, depth(128), [1, 7])
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0d_7x1' % m6b, depth(128), [7, 1])
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0e_1x7' % m6b, depth(192), [1, 7])
    br3 = net.add(AvgPooling2D('%s/Branch_3/AvgPool_0a_3x3' % m6b, 3, 1), s)
    br3 = conv2d(net, '%s/Branch_3/Conv2d_0b_1x1' % m6b, depth(192), [1, 1])
    endpoints[m6b] = net.add(Concat('%s/Concat' % m6b, 1),  [br0, br1, br2, br3])
    final_aux_check(m6b, net)

    # mixed_5: 17 x 17 x 768.
    m6c = V3 + '/Mixed_6c'
    s = net.add(Split('%s/Split' % m6c, 4))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_0a_1x1' % m6c, depth(192), [1, 1], src=s)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % m6c, depth(160), [1, 1], src=s)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0b_1x7' % m6c, depth(160), [1, 7])
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0c_7x1' % m6c, depth(192), [7, 1])
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0a_1x1' % m6c, depth(160), [1, 1], src=s)
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0b_7x1' % m6c, depth(160), [7, 1])
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0c_1x7' % m6c, depth(160), [1, 7])
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0d_7x1' % m6c, depth(160), [7, 1])
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0e_1x7' % m6c, depth(192), [1, 7])
    br3 = net.add(AvgPooling2D('%s/Branch_3/AvgPool_0a_3x3' % m6c, 3, 1), s)
    br3 = conv2d(net, '%s/Branch_3/Conv2d_0b_1x1' % m6c, depth(192), [1, 1])
    endpoints[m6c] = net.add(Concat('%s/Concat' % m6c, 1),  [br0, br1, br2, br3])
    final_aux_check(m6c, net)

    # mixed_6: 17 x 17 x 768.
    m6d = V3 + '/Mixed_6d'
    s = net.add(Split('%s/Split' % m6d, 4))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_0a_1x1' % m6d, depth(192), [1, 1], src=s)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % m6d, depth(160), [1, 1], src=s)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0b_1x7' % m6d, depth(160), [1, 7])
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0c_7x1' % m6d, depth(192), [7, 1])
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0a_1x1' % m6d, depth(160), [1, 1], src=s)
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0b_7x1' % m6d, depth(160), [7, 1])
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0c_1x7' % m6d, depth(160), [1, 7])
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0d_7x1' % m6d, depth(160), [7, 1])
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0e_1x7' % m6d, depth(192), [1, 7])
    br3 = net.add(AvgPooling2D('%s/Branch_3/AvgPool_0a_3x3' % m6d, 3, 1), s)
    br3 = conv2d(net, '%s/Branch_3/Conv2d_0b_1x1' % m6d, depth(192), [1, 1])
    endpoints[m6d] = net.add(Concat('%s/Concat' % m6d, 1),  [br0, br1, br2, br3])
    final_aux_check(m6d, net)

    m6e = V3 + '/Mixed_6e'
    s = net.add(Split('%s/Split' % m6e, 4))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_0a_1x1' % m6e, depth(192), [1, 1], src=s)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % m6e, depth(192), [1, 1], src=s)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0b_1x7' % m6e, depth(192), [1, 7])
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0c_7x1' % m6e, depth(192), [7, 1])
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0a_1x1' % m6e, depth(192), [1, 1], src=s)
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0b_7x1' % m6e, depth(192), [7, 1])
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0c_1x7' % m6e, depth(192), [1, 7])
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0d_7x1' % m6e, depth(192), [7, 1])
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0e_1x7' % m6e, depth(192), [1, 7])
    br3 = net.add(AvgPooling2D('%s/Branch_3/AvgPool_0a_3x3' % m6d, 3, 1), s)
    br3 = conv2d(net, '%s/Branch_3/Conv2d_0b_1x1' % m6d, depth(192), [1, 1])
    endpoints[m6e] = net.add(Concat('%s/Concat' % m6d, 1),  [br0, br1, br2, br3])
    final_aux_check(m6e, net)

    # mixed_8: 8 x 8 x 1280.
    m7a = V3 + '/Mixed_7a'
    s = net.add(Split('%s/Split' % m7a, 3))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_0a_1x1' % m7a, depth(192), [1, 1], src=s)
    br0 = conv2d(net, '%s/Branch_0/Conv2d_1a_3x3' % m7a, depth(320), [3, 3], 2, border_mode='VALID')
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % m7a, depth(192), [1, 1], src=s)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0b_1x7' % m7a, depth(192), [1, 7])
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0c_7x1' % m7a, depth(192), [7, 1])
    br1 = conv2d(net, '%s/Branch_1/Conv2d_1a_3x3' % m7a, depth(192), [3, 3], 2, border_mode='VALID')
    br2 = net.add(MaxPooling2D('%s/Branch_2/MaxPool_1a_3x3' % m7a, 3, 2, border_mode='VALID'), s)
    endpoints[m7a] = net.add(Concat('%s/Concat' % m7a, 1),  [br0, br1, br2])
    final_aux_check(m7a, net)

    # mixed_9: 8 x 8 x 2048.
    m7b = V3 + '/Mixed_7b'
    s = net.add(Split('%s/Split' % m7b, 4))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_0a_1x1' % m7b, depth(320), [1, 1], src=s)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % m7b, depth(384), [1, 1], src=s)
    s1 = net.add(Split('%s/Branch_1/Split1' % m7b, 2))
    br11 = conv2d(net, '%s/Branch_1/Conv2d_0b_1x3' % m7b, depth(384), [1, 3], src=s1)
    br12 = conv2d(net, '%s/Branch_1/Conv2d_0b_3x1' % m7b, depth(384), [3, 1], src=s1)
    br1 = net.add(Concat('%s/Branch_1/Concat1' % m7b, 1),  [br11, br12])
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0a_1x1' % m7b, depth(448), [1, 1], src=s)
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0b_3x3' % m7b, depth(384), [3, 3])
    s2 = net.add(Split('%s/Branch_2/Split2' % m7b, 2))
    br21 = conv2d(net, '%s/Branch_2/Conv2d_0c_1x3' % m7b, depth(384), [1, 3], src=s2)
    br22 = conv2d(net, '%s/Branch_2/Conv2d_0d_3x1' % m7b, depth(384), [3, 1], src=s2)
    br2 = net.add(Concat('%s/Branch_2/Concat2' % m7b, 1),  [br21, br22])
    br3 = net.add(AvgPooling2D('%s/Branch_3/AvgPool_0a_3x3' % m7b, 3, 1), src=s)
    br3 = conv2d(net, '%s/Branch_3/Conv2d_0b_1x1' % m7b, depth(192), [1, 1])
    endpoints[m7b] = net.add(Concat('%s/Concat' % m7b, 1),  [br0, br1, br2, br3])
    final_aux_check(m7b, net)

    # mixed_10: 8 x 8 x 2048.
    m7c = V3 + '/Mixed_7c'
    s = net.add(Split('%s/Split' % m7c, 4))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_0a_1x1' % m7c, depth(320), [1, 1], src=s)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % m7c, depth(384), [1, 1], src=s)
    s1 = net.add(Split('%s/Branch_1/Split1' % m7c, 2))
    br11 = conv2d(net, '%s/Branch_1/Conv2d_0b_1x3' % m7c, depth(384), [1, 3], src=s1)
    br12 = conv2d(net, '%s/Branch_1/Conv2d_0b_3x1' % m7c, depth(384), [3, 1], src=s1)
    br1 = net.add(Concat('%s/Branch_1/Concat1' % m7c, 1),  [br11, br12])
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0a_1x1' % m7c, depth(448), [1, 1], src=s)
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0b_3x3' % m7c, depth(384), [3, 3])
    s2 = net.add(Split('%s/Branch_2/Split2' % m7c, 2))
    br21 = conv2d(net, '%s/Branch_2/Conv2d_0c_1x3' % m7c, depth(384), [1, 3], src=s2)
    br22 = conv2d(net, '%s/Branch_2/Conv2d_0d_3x1' % m7c, depth(384), [3, 1], src=s2)
    br2 = net.add(Concat('%s/Branch_2/Concat2' % m7c, 1),  [br21, br22])
    br3 = net.add(AvgPooling2D('%s/Branch_3/AvgPool_0a_3x3' % m7c, 3, 1), src=s)
    br3 = conv2d(net, '%s/Branch_3/Conv2d_0b_1x1' % m7c, depth(192), [1, 1])
    endpoints[m7c] = net.add(Concat('%s/Concat' % m7c, 1),  [br0, br1, br2, br3])
    final_aux_check(m7c, net)
    return net, endpoints[m7c], endpoints


def create_net(num_classes=1001, sample_shape=(3, 299, 299), is_training=True, dropout_keep_prob=0.8, create_aux_logits=True):
    """Creates the Inception V4 model.

    Args:
        num_classes: number of predicted classes.
        is_training: whether is training or not.
        dropout_keep_prob: float, the fraction to keep before final layer.
        reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
        create_aux_logits: Whether to include the auxiliary logits.

    Returns:
        logits: the logits outputs of the model.
        end_points: the set of end_points from the inception model.
    """
    name = 'InceptionV3'
    if is_training and create_aux_logits:
        aux_name = name + '/Mixed_6e'
    else:
        aux_name = None
    net, last_layer, end_points = inception_v3_base(name, sample_shape, aux_name=aux_name)
    # Auxiliary Head logits
    if aux_name is not None:
        # 8 x 8 x 1280
        aux_logits = end_points[aux_name + '-aux']
        net.add(AvgPooling2D('%s/AuxLogits/AvgPool_1a_5x5' % name, 5, stride=3, border_mode='VALID'), aux_logits)
        t = conv2d(net, '%s/AuxLogits/Conv2d_1b_1x1' % name, 128, 1)
        conv2d(net, '%s/AuxLogits/Conv2d_2a' % name, 768, t.get_output_sample_shape()[1:3], border_mode='VALID')
        net.add(Flatten('%s/AuxLogits/flat' % name))
        end_points['AuxLogits'] = net.add(Dense('%s/AuxLogits/Aux_logits' % name, num_classes))

    # Final pooling and prediction
    # 8 x 8 x 2048
    net.add(AvgPooling2D('%s/Logits/AvgPool_1a' % name, last_layer.get_output_sample_shape()[1:3], 1, border_mode='VALID'), last_layer)
    # 1 x 1 x 2048
    net.add(Dropout('%s/Logits/Dropout_1b' % name, 1 - dropout_keep_prob))
    net.add(Flatten('%s/Logits/PreLogitsFlatten' % name))
    # 2048
    end_points['Logits'] = net.add(Dense('%s/Logits/Logits' % name, num_classes))
    return net, end_points


if __name__ == '__main__':
    net, _ = create_net()
