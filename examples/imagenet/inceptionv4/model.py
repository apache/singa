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

def conv2d(net, name, nb_filter, k, s=1, padding='SAME', src=None):
    net.add(Conv2D(name, nb_filter, k, s, border_mode=padding, use_bias=False), src)
    net.add(BatchNormalization('%s/BatchNorm' % name))
    return net.add(Activation(name+'/relu'))


def block_inception_a(name, net):
    """Builds Inception-A block for Inception v4 network."""
    # By default use stride=1 and SAME padding
    split = net.add(Split('%s/Split' % name, 4))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_0a_1x1' % name, 96, 1, src=split)
    conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % name, 64, 1, src=split)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0b_3x3' % name, 96, 3)
    conv2d(net, '%s/Branch_2/Conv2d_0a_1x1' % name, 64, 1, src=split)
    conv2d(net, '%s/Branch_2/Conv2d_0b_3x3' % name, 96, 3)
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0c_3x3' % name, 96, 3)
    net.add(AvgPooling2D('%s/Branch_3/AvgPool_0a_3x3' % name, 3, stride=1), split)
    br3 = conv2d(net, '%s/Branch_3/Conv2d_0b_1x1' % name, 96, 1)
    return net.add(Concat('%s/Concat' % name, 1), [br0, br1, br2, br3])


def block_reduction_a(name, net):
    """Builds Reduction-A block for Inception v4 network."""
    # By default use stride=1 and SAME padding
    split = net.add(Split('%s/Split' % name, 3))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_1a_3x3' % name, 384, 3, 2, padding='VALID', src=split)
    conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % name, 192, 1, src=split)
    conv2d(net, '%s/Branch_1/Conv2d_0b_3x3' % name, 224, 3)
    br1 = conv2d(net, '%s/Branch_1/Conv2d_1a_3x3' % name, 256, 3, 2, padding='VALID')
    br2 = net.add(MaxPooling2D('%s/Branch_2/MaxPool_1a_3x3' % name, 3, 2, border_mode='VALID'), split)
    return net.add(Concat('%s/Concat' % name, 1), [br0, br1, br2])


def block_inception_b(name, net):
    """Builds Inception-B block for Inception v4 network."""
    # By default use stride=1 and SAME padding
    split = net.add(Split('%s/Split' % name, 4))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_0a_1x1' % name, 384, 1, src=split)
    conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % name, 192, 1, src=split)
    conv2d(net, '%s/Branch_1/Conv2d_0b_1x7' % name, 224, (1, 7))
    br1 = conv2d(net, '%s/Branch_1/Conv2d_0c_7x1' % name, 256, (7, 1))
    conv2d(net, '%s/Branch_2/Conv2d_0a_1x1' % name, 192, 1, src=split)
    conv2d(net, '%s/Branch_2/Conv2d_0b_7x1' % name, 192, (7, 1))
    conv2d(net, '%s/Branch_2/Conv2d_0c_1x7' % name, 224, (1, 7))
    conv2d(net, '%s/Branch_2/Conv2d_0d_7x1' % name, 224, (7, 1))
    br2 = conv2d(net, '%s/Branch_2/Conv2d_0e_1x7' % name, 256, (1, 7))
    net.add(AvgPooling2D('%s/Branch_3/AvgPool_0a_3x3' % name, 3, 1), split)
    br3 = conv2d(net, '%s/Branch_3/Conv2d_0b_1x1' % name, 128, 1)
    return net.add(Concat('%s/Concat' % name, 1), [br0, br1, br2, br3])


def block_reduction_b(name, net):
    """Builds Reduction-B block for Inception v4 network."""
    # By default use stride=1 and SAME padding
    split = net.add(Split('%s/Split', 3))
    conv2d(net, '%s/Branch_0/Conv2d_0a_1x1' % name, 192, 1, src=split)
    br0 = conv2d(net, '%s/Branch_0/Conv2d_1a_3x3' % name, 192, 3, 2, padding='VALID')
    conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % name, 256, 1, src=split)
    conv2d(net, '%s/Branch_1/Conv2d_0b_1x7' % name, 256, (1, 7))
    conv2d(net, '%s/Branch_1/Conv2d_0c_7x1' % name, 320, (7, 1))
    br1 = conv2d(net, '%s/Branch_1/Conv2d_1a_3x3' % name, 320, 3, 2, padding='VALID')
    br2 = net.add(MaxPooling2D('%s/Branch_2/MaxPool_1a_3x3' % name, 3, 2, border_mode='VALID'), split)
    return net.add(Concat('%s/Concat' % name, 1), [br0, br1, br2])


def block_inception_c(name, net):
    """Builds Inception-C block for Inception v4 network."""
    # By default use stride=1 and SAME padding
    split = net.add(Split('%s/Split' % name, 4))
    br0 = conv2d(net, '%s/Branch_0/Conv2d_0a_1x1' % name, 256, 1, src=split)
    conv2d(net, '%s/Branch_1/Conv2d_0a_1x1' % name, 384, 1, src=split)
    br1_split = net.add(Split('%s/Branch_1/Split' % name, 2))
    br1_0 = conv2d(net, '%s/Branch_1/Conv2d_0b_1x3' % name, 256, (1, 3), src=br1_split)
    br1_1 = conv2d(net, '%s/Branch_1/Conv2d_0c_3x1' % name, 256, (3, 1), src=br1_split)
    br1 = net.add(Concat('%s/Branch_1/Concat' % name, 1), [br1_0, br1_1])
    conv2d(net, '%s/Branch_2/Conv2d_0a_1x1' % name, 384, 1, src=split)
    conv2d(net, '%s/Branch_2/Conv2d_0b_3x1' % name, 448, (3, 1))
    conv2d(net, '%s/Branch_2/Conv2d_0c_1x3' % name, 512, (1, 3))
    br2_split = net.add(Split('%s/Branch_2/Split' % name, 2))
    br2_0 = conv2d(net, '%s/Branch_2/Conv2d_0d_1x3' % name, 256, (1, 3), src=br2_split)
    br2_1 = conv2d(net, '%s/Branch_2/Conv2d_0e_3x1' % name, 256, (3, 1), src=br2_split)
    br2 = net.add(Concat('%s/Branch_2/Concat' % name, 1), [br2_0, br2_1])
    net.add(AvgPooling2D('%s/Branch_3/AvgPool_0a_3x3' % name, 3, 1), split)
    br3 = conv2d(net, '%s/Branch_3/Conv2d_0b_1x1' % name, 256, 1)
    return net.add(Concat('%s/Concat' % name, 1), [br0, br1, br2, br3])


def inception_v4_base(name, sample_shape, final_endpoint='Mixed_7d', aux_name=None):
    """Creates the Inception V4 network up to the given final endpoint.

    Args:
        inputs: a 4-D tensor of size [batch_size, height, width, 3].
        final_endpoint: specifies the endpoint to construct the network up to.
        It can be one of [ 'Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
        'Mixed_3a', 'Mixed_4a', 'Mixed_5a', 'Mixed_5b', 'Mixed_5c', 'Mixed_5d',
        'Mixed_5e', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c', 'Mixed_6d', 'Mixed_6e',
        'Mixed_6f', 'Mixed_6g', 'Mixed_6h', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c',
        'Mixed_7d']

    Returns:
        logits: the logits outputs of the model.
        end_points: the set of end_points from the inception model.

    Raises:
        ValueError: if final_endpoint is not set to one of the predefined values,
    """
    end_points = {}
    net = ffnet.FeedForwardNet()
    def add_and_check_final(name, lyr):
        end_points[name] = lyr
        return name == final_endpoint

    # 299 x 299 x 3
    net.add(Conv2D('%s/Conv2d_1a_3x3' % name, 32, 3, 2, border_mode='VALID', use_bias=False, input_sample_shape=sample_shape))
    net.add(BatchNormalization('%s/Conv2d_1a_3x3/BatchNorm' % name))
    net.add(Activation('%s/Conv2d_1a_3x3/relu' % name))
    # 149 x 149 x 32
    conv2d(net, '%s/Conv2d_2a_3x3' % name, 32, 3, padding='VALID')
    # 147 x 147 x 32
    conv2d(net, '%s/Conv2d_2b_3x3' % name, 64, 3)
    # 147 x 147 x 64
    s = net.add(Split('%s/Mixed_3a/Split' % name, 2))
    br0 = net.add(MaxPooling2D('%s/Mixed_3a/Branch_0/MaxPool_0a_3x3' % name, 3, 2, border_mode='VALID'), s)
    br1 = conv2d(net, '%s/Mixed_3a/Branch_1/Conv2d_0a_3x3' % name, 96, 3, 2, padding='VALID', src=s)
    net.add(Concat('%s/Mixed_3a/Concat' % name, 1), [br0, br1])

    # 73 x 73 x 160
    s = net.add(Split('%s/Mixed_4a/Split' % name, 2))
    conv2d(net, '%s/Mixed_4a/Branch_0/Conv2d_0a_1x1' % name, 64, 1, src=s)
    br0 = conv2d(net, '%s/Mixed_4a/Branch_0/Conv2d_1a_3x3' % name, 96, 3, padding='VALID')
    conv2d(net, '%s/Mixed_4a/Branch_1/Conv2d_0a_1x1' % name, 64, 1, src=s)
    conv2d(net, '%s/Mixed_4a/Branch_1/Conv2d_0b_1x7' % name, 64, (1, 7))
    conv2d(net, '%s/Mixed_4a/Branch_1/Conv2d_0c_7x1' % name, 64, (7, 1))
    br1 = conv2d(net, '%s/Mixed_4a/Branch_1/Conv2d_1a_3x3' % name, 96, 3, padding='VALID')
    net.add(Concat('%s/Mixed_4a/Concat' % name, 1), [br0, br1])

      # 71 x 71 x 192
    s = net.add(Split('%s/Mixed_5a/Split' % name, 2))
    br0 = conv2d(net, '%s/Mixed_5a/Branch_0/Conv2d_1a_3x3' % name, 192, 3, 2, padding='VALID', src=s)
    br1 = net.add(MaxPooling2D('%s/Mixed_5a/Branch_1/MaxPool_1a_3x3' % name, 3, 2, border_mode='VALID'), s)
    net.add(Concat('%s/Mixed_5a/Concat' % name, 1), [br0, br1])

    # 35 x 35 x 384
    # 4 x Inception-A blocks
    for idx in range(4):
        block_scope = name + '/Mixed_5' + chr(ord('b') + idx)
        lyr = block_inception_a(block_scope, net)
        if add_and_check_final(block_scope, lyr): return net, lyr, end_points

    # 35 x 35 x 384
    # Reduction-A block
    block_reduction_a(name + '/Mixed_6a', net)

    # 17 x 17 x 1024
    # 7 x Inception-B blocks
    for idx in range(7):
        block_scope = name + '/Mixed_6' + chr(ord('b') + idx)
        lyr = block_inception_b(block_scope, net)
        if add_and_check_final(block_scope, lyr): return net, lyr, end_points
        if block_scope == aux_name:
            end_points[aux_name] = net.add(Split('%s/Split' % block_scope, 2))

    # 17 x 17 x 1024
    # Reduction-B block
    block_reduction_b(name + '/Mixed_7a', net)

    # 8 x 8 x 1536
    # 3 x Inception-C blocks
    for idx in range(3):
        block_scope = name + '/Mixed_7' + chr(ord('b') + idx)
        lyr = block_inception_c(block_scope, net)
        if add_and_check_final(block_scope, lyr): return net, lyr, end_points
        if block_scope == aux_name:
            end_points[aux_name] = net.add(Split('%s/Split' % block_scope, 2))
    return net, lyr, end_points


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
    end_points = {}
    name = 'InceptionV4'
    if is_training and create_aux_logits:
        aux_name = name + '/Mixed_6h'
    else:
        aux_name = None
    net, last_layer, end_points = inception_v4_base(name, sample_shape, aux_name=aux_name)
    # Auxiliary Head logits
    if aux_name is not None:
        # 17 x 17 x 1024
        aux_logits = end_points[aux_name]
        net.add(AvgPooling2D('%s/AuxLogits/AvgPool_1a_5x5' % name, 5, stride=3, border_mode='VALID'), aux_logits)
        t = conv2d(net, '%s/AuxLogits/Conv2d_1b_1x1' % name, 128, 1)
        conv2d(net, '%s/AuxLogits/Conv2d_2a' % name, 768, t.get_output_sample_shape()[1:3], padding='VALID')
        net.add(Flatten('%s/AuxLogits/flat' % name))
        end_points['AuxLogits'] = net.add(Dense('%s/AuxLogits/Aux_logits' % name, num_classes))

    # Final pooling and prediction
    # 8 x 8 x 1536
    net.add(AvgPooling2D('%s/Logits/AvgPool_1a' % name, last_layer.get_output_sample_shape()[1:3], border_mode='VALID'), last_layer)
    # 1 x 1 x 1536
    net.add(Dropout('%s/Logits/Dropout_1b' % name, 1 - dropout_keep_prob))
    net.add(Flatten('%s/Logits/PreLogitsFlatten' % name))
    # 1536
    end_points['Logits'] = net.add(Dense('%s/Logits/Logits' % name, num_classes))
    return net, end_points


if __name__ == '__main__':
    net, _ = create_net()
