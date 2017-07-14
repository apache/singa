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
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# =============================================================================
'''Popular initialization methods for parameter values (Tensor objects).

Example usages::

    from singa import tensor
    from singa import initializer

    x = tensor.Tensor((3, 5))
    initializer.uniform(x, 3, 5) # use both fan_in and fan_out
    initializer.uniform(x, 3, 0)  # use only fan_in
'''
from __future__ import division
import math


def uniform(t, fan_in=0, fan_out=0):
    '''Initialize the values of the input tensor following a uniform
    distribution with specific bounds.

    Args:
        fan_in(int): for the weight Tensor of a convolution layer,
            fan_in = nb_channel * kh * kw; for dense layer,
            fan_in = input_feature_length
        fan_out(int): for the convolution layer weight Tensor,
            fan_out = nb_filter * kh * kw; for the weight Tensor of a dense
            layer, fan_out = output_feature_length

    Ref: [Bengio and Glorot 2010]: Understanding the difficulty of
    training deep feedforward neuralnetworks.

    '''
    assert fan_in > 0 or fan_out > 0, \
        'fan_in and fan_out cannot be 0 at the same time'
    avg = 2
    if fan_in * fan_out == 0:
        avg = 1
    x = math.sqrt(3.0 * avg / (fan_in + fan_out))
    t.uniform(-x, x)


def gaussian(t, fan_in=0, fan_out=0):
    '''Initialize the values of the input tensor following a Gaussian
    distribution with specific std.

    Args:
        fan_in(int): for the weight Tensor of a convolution layer,
            fan_in = nb_channel * kh * kw; for dense layer,
            fan_in = input_feature_length
        fan_out(int): for the convolution layer weight Tensor,
            fan_out = nb_filter * kh * kw; for the weight Tensor of a dense
            layer, fan_out = output_feature_length

    Ref Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun: Delving Deep into
    Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
    '''
    assert fan_in > 0 or fan_out > 0, \
        'fan_in and fan_out cannot be 0 at the same time'
    avg = 2
    if fan_in * fan_out == 0:
        avg = 1
    std = math.sqrt(2.0 * avg / (fan_in + fan_out))
    t.gaussian(0, std)


def xavier(t):
    '''Initialize the matrix parameter follow a Uniform distribution from
    [-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out))].

    Deprecated. Please use uniform()

    Args:
        t (Tensor): the parater tensor
    '''

    scale = math.sqrt(6.0 / (t.shape[0] + t.shape[1]))
    t.uniform(-scale, scale)


def glorot(t):
    '''Initialize the matrix parameter follow a Gaussian distribution with
    mean = 0 and std = sqrt(2.0 / (nb_row + nb_col))

    Deprecated. Please use gaussian()

    Args:
        t (Tensor): the parater tensor
    '''
    scale = math.sqrt(2.0 / (t.shape[0] + t.shape[1]))
    t.gaussian(0, 1)
    t *= scale


def msra(t):
    '''Initialize the matrix parameter follow a Guassian distribution with
    mean = 0, std = math.sqrt(2.0 / nb_row).

    Deprecated. Please use gaussian()

    Ref [He, Zhang, Ren and Sun 2015]: Specifically accounts for ReLU
    nonlinearities.

    Args:
        t (Tensor): the parater tensor
    '''
    t.gaussian(0, math.sqrt(2.0 / t.shape[0]))
