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
    initializer.xavier(x)
'''

import math


'''
TODO(wangwei) update the uniform and gaussian initializers

def uniform(t, fan_in=0, fan_out=0):
    typically, for conv layer weight: fan_in = nb_filter * kh * kw,
    fan_out = nb_channel * kh * kw
    for dense layer weight, fan_in = input_feature_length,
    fan_out = output_feature_length
    # Ref: [Bengio and Glorot 2010]: Understanding the difficulty of
    training deep feedforward neuralnetworks.

    assert fan_in >0 or fan_out > 0, \
        'fan_in and fan_out cannot be 0 at the same time'
    avg = 1
    if fan_in * fan_out == 0:
      avg = 2
    x = math.sqrt(3.0f * avg / (fan_in + fan_out))
    t.uniform(-x, x)


def gaussian(t, fan_in=0, fan_out=0):
    typically, for conv layer weight: fan_in = nb_filter * kh * kw,
    fan_out = nb_channel * kh * kw
    for dense layer weight, fan_in = input_feature_length,
    fan_out = output_feature_length

    Ref Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun: Delving Deep into
    Rectifiers: Surpassing Human-Level Performance on ImageNet Classification

    assert fan_in >0 or fan_out > 0, \
        'fan_in and fan_out cannot be 0 at the same time'
    avg = 1
    if fan_in * fan_out == 0:
      avg = 2
    std = math.sqrt(2.0f * avg / (fan_in + fan_out))
    t.gaussian(0, std)
'''


def uniform(t, low=0, high=1):
    '''Initialize the parameter values following an Uniform distribution.

    Args:
        t (Tensor): the parater tensor
        low (float): lower bound
        high (float): higher bound
    '''
    t.uniform(low, high)


def gaussian(t, mean=0, std=0.01):
    '''Initialize the parameter values following an Gaussian distribution.

    Args:
        t (Tensor): the parater tensor
        mean (float): mean of the distribution
        std (float): standard variance
    '''
    t.gaussian(mean, std)


def xavier(t):
    '''Initialize the matrix parameter follow a Uniform distribution from
    [-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out))].

    Args:
        t (Tensor): the parater tensor
    '''

    scale = math.sqrt(6.0 / (t.shape[0] + t.shape[1]))
    t.uniform(-scale, scale)


def glorot(t):
    '''Initialize the matrix parameter follow a Gaussian distribution with
    mean = 0 and std = sqrt(2.0 / (nb_row + nb_col))

    Args:
        t (Tensor): the parater tensor
    '''
    scale = math.sqrt(2.0 / (t.shape[0] + t.shape[1]))
    t.gaussian(0, 1)
    t *= scale


def msra(t):
    '''Initialize the matrix parameter follow a Guassian distribution with
    mean = 0, std = math.sqrt(2.0 / nb_row).

    Ref [He, Zhang, Ren and Sun 2015]: Specifically accounts for ReLU
    nonlinearities.

    Args:
        t (Tensor): the parater tensor
    '''
    t.gaussian(0, math.sqrt(2.0 / t.shape[0]))
