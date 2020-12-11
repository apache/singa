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

credit: this module is adapted from keras
https://github.com/keras-team/keras/blob/master/keras/initializers.py

All functions in this module change the input tensor in-place.

Example usages::

    from singa import tensor
    from singa import initializer

    x = tensor.Tensor((3, 5))
    initializer.he_uniform(x)
    initializer.golorot_norm(x) 
'''

from __future__ import division
import math
import numpy as np
from deprecated import deprecated


def eye(t):
    """Initialize the tensor with ones on the diagonal and zeros elsewhere.

    Note: it is implemented by calling numpy. 
    Do not call it within forward propagation when computation graph is enabled.

    # Arguments
        t(Tensor): the matrix to be filled in.
    """
    if len(t.shape) == 2:
        raise ValueError("Only tensors with 2 dimensions are supported")
    a = np.eye(t.shape[0], t.shape[1], dtype=np.float32)
    t.copy_from(a)


def orthogonal(t, gain=1.0):
    """Initializer that generates a random orthogonal matrix.

    Note: it is implemented by calling numpy. 
    Do not call it within forward propagation when computation graph is enabled.

    # Arguments
        t(Tensor): the matrix to be filled in.
        gain: Multiplicative factor to apply to the orthogonal matrix.

    # References
        - [Exact solutions to the nonlinear dynamics of learning in deep
           linear neural networks](http://arxiv.org/abs/1312.6120)
    """
    if len(t.shape) == 2:
        raise ValueError("Only tensors with 2 dimensions are supported")

    a = np.random.normal(0.0, 1.0, t.shape).astype(np.float32)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    # Pick the one with the correct shape.
    q = u if u.shape == t.shape else v
    q *= gain
    t.copy_from(q)


def lecun_uniform(t):
    """LeCun uniform initializer.

    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(3 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.

    # Arguments
        t(Tensor):the tensor to be filled in.

    # References
        - [Efficient BackProp](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
    """
    _random_fill(t, scale=1., mode='fan_in', distribution='uniform')


def glorot_normal(t):
    """Glorot normal initializer, also called Xavier normal initializer.

    It draws samples from a normal distribution centered on 0
    with `stddev = sqrt(2 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor
    and `fan_out` is the number of output units in the weight tensor.

    # Arguments
        t(Tensor):the tensor to be filled in.

    # References
        - [Understanding the difficulty of training deep feedforward neural
           networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
    """
    _random_fill(t, scale=1., mode='fan_avg', distribution='normal')


def glorot_uniform(t):
    """Glorot uniform initializer, also called Xavier uniform initializer.

    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(6 / (fan_in + fan_out))`
    where `fan_in` is the number of input units in the weight tensor
    and `fan_out` is the number of output units in the weight tensor.

    # Arguments
        t(Tensor):the tensor to be filled in.
    # References
        - [Understanding the difficulty of training deep feedforward neural
           networks](http://jmlr.org/proceedings/papers/v9/glorot10a/glorot10a.pdf)
    """
    _random_fill(t, scale=1., mode='fan_avg', distribution='uniform')


def he_normal(t):
    """He normal initializer.

    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(2 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.

    # Arguments
        t(Tensor):the tensor to be filled in.

    # References
        - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on
           ImageNet Classification](http://arxiv.org/abs/1502.01852)
    """
    _random_fill(t, scale=2., mode='fan_in', distribution='normal')

def lecun_normal(t):
    """LeCun normal initializer.

    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(1 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.

    # Arguments
        t(Tensor):the tensor to be filled in.

    # References
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
        - [Efficient Backprop](http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
    """
    _random_fill(t, scale=1., mode='fan_in', distribution='normal')


def he_uniform(t):
    '''Initialize the values of the input tensor following a uniform
    distribution with specific bounds.

    It draws samples from a uniform distribution within [-limit, limit]
    where `limit` is `sqrt(6 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.

    # Arguments
        t(Tensor): the tensor to be filled in.

    # References
        - [Delving Deep into Rectifiers: Surpassing Human-Level Performance on
           ImageNet Classification](http://arxiv.org/abs/1502.01852)
    '''
    _random_fill(t, scale=2., mode='fan_in', distribution='uniform')


@deprecated(reason="Use he_normal or glorot_normal")
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


@deprecated(reason="Use glorot_normal")
def xavier(t):
    '''Initialize the matrix parameter follow a Uniform distribution from
    [-sqrt(6/(fan_in + fan_out)), sqrt(6/(fan_in + fan_out))].

    Args:
        t (Tensor): the parater tensor
    '''

    scale = math.sqrt(6.0 / (t.shape[0] + t.shape[1]))
    t.uniform(-scale, scale)


@deprecated(reason="Use glorot_uniform")
def glorot(t):
    '''Initialize the matrix parameter follow a Gaussian distribution with
    mean = 0 and std = sqrt(2.0 / (nb_row + nb_col))

    Args:
        t (Tensor): the parater tensor
    '''
    scale = math.sqrt(2.0 / (t.shape[0] + t.shape[1]))
    t.gaussian(0, 1)
    t *= scale


@deprecated(reason="Use he_normal")
def msra(t):
    '''Initialize the matrix parameter follow a Guassian distribution with
    mean = 0, std = math.sqrt(2.0 / nb_row).

    Ref [He, Zhang, Ren and Sun 2015]: Specifically accounts for ReLU
    nonlinearities.

    Args:
        t (Tensor): the parater tensor
    '''
    t.gaussian(0, math.sqrt(2.0 / t.shape[0]))


def _compute_fans(shape, data_format='channels_first'):
    """Computes the number of input and output units for a weight shape.
    # Arguments
        shape: Integer shape tuple.
        data_format: Image data format to use for convolution kernels.
            Note that all kernels in Keras are standardized on the
            `channels_last` ordering (even when inputs are set
            to `channels_first`).
    # Returns
        A tuple of scalars, `(fan_in, fan_out)`.
    # Raises
        ValueError: in case of invalid `data_format` argument.
    """
    if len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    elif len(shape) in {3, 4, 5}:
        # Assuming convolution kernels (1D, 2D or 3D).
        # TH kernel shape: (depth, input_depth, ...)
        # TF kernel shape: (..., input_depth, depth)
        if data_format == 'channels_first':
            receptive_field_size = np.prod(shape[2:])
            fan_in = shape[1] * receptive_field_size
            fan_out = shape[0] * receptive_field_size
        elif data_format == 'channels_last':
            receptive_field_size = np.prod(shape[:-2])
            fan_in = shape[-2] * receptive_field_size
            fan_out = shape[-1] * receptive_field_size
        else:
            raise ValueError('Invalid data_format: ' + data_format)
    else:
        # No specific assumptions.
        fan_in = np.sqrt(np.prod(shape))
        fan_out = np.sqrt(np.prod(shape))
    return fan_in, fan_out


def _random_fill(t, scale, mode, distribution):
    """Fill the tensor with values sampled from a distribution.

    With `distribution="normal"`, samples are drawn from a normal
    distribution centered on zero, with `stddev = sqrt(scale / n)` where n is:
        - number of input units in the weight tensor, if mode = "fan_in"
        - number of output units, if mode = "fan_out"
        - average of the numbers of input and output units, if mode = "fan_avg"

    With `distribution="uniform"`,
    samples are drawn from a uniform distribution
    within [-limit, limit], with `limit = sqrt(3 * scale / n)`.


    Args:
        t (Tensor): Tensor to be filled
        scale (float): scale factor  
        mode (str): "fan_in" or "fan_out" or "fan_avg" 
        distribution (str): "normal" or "uniform" 

    Raises:
        ValueError: In case of an invalid value for scale, mode or distribution 
    """
    if scale <= 0.:
        raise ValueError('`scale` must be a positive float. Got:', scale)
    mode = mode.lower()
    if mode not in {'fan_in', 'fan_out', 'fan_avg'}:
        raise ValueError(
            'Invalid `mode` argument: '
            'expected on of {"fan_in", "fan_out", "fan_avg"} '
            'but got', mode)
    distribution = distribution.lower()
    if distribution not in {'normal', 'uniform'}:
        raise ValueError(
            'Invalid `distribution` argument: '
            'expected one of {"normal", "uniform"} '
            'but got', distribution)

    fan_in, fan_out = _compute_fans(t.shape)
    if mode == 'fan_in':
        scale /= max(1., fan_in)
    elif mode == 'fan_out':
        scale /= max(1., fan_out)
    else:
        scale /= max(1., float(fan_in + fan_out) / 2)
    if distribution == 'normal':
        # 0.879... = scipy.stats.truncnorm.std(a=-2, b=2, loc=0., scale=1.)
        # stddev = np.sqrt(scale) / .87962566103423978
        t.gaussian(0., np.sqrt(scale))
    else:
        limit = np.sqrt(3. * scale)
        t.uniform(-limit, limit)