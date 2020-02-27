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

import sys
import math
import numpy as np

from singa import tensor
from . import singa_wrap as singa

def update_progress(progress, info):
    """Display progress bar and user info.

    Args:
        progress (float): progress [0, 1], negative for halt, and >=1 for done.
        info (str): a string for user provided info to be displayed.
    """
    barLength = 20  # bar length
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float. "
    if progress < 0:
        progress = 0
        status = "Halt. "
    if progress >= 1:
        progress = 1
        status = "Done. "
    status = status + info
    block = int(round(barLength * progress))
    text = "[{0}] {1:3.1f}% {2}".format("." * block + " " * (barLength - block),
                                        progress * 100, status)
    sys.stdout.write(text)
    sys.stdout.write('\b' * (9 + barLength + len(status)))
    sys.stdout.flush()


def handle_same_pad_fwd(x, padding, pad_mode):
    """
    handle same padding mode forward
    Args:x
        the input tensor
    Args:padding
        the padding
    Returns: 
        tensor, the output
    """
    # In case of odd number add the extra padding at the end for SAME_UPPER
    # at the beginning for SAME_LOWER
    x_tensor = tensor.from_raw_tensor(x)
    for axis, pad in zip((2, 3), padding):
        if pad % 2 != 0:
            zeros_shape = list(x_tensor.data.shape())
            zeros_shape[axis] = 1
            zero_padding = np.zeros(zeros_shape).astype(np.float32)
            zero_padding = tensor.Tensor(device=x.device(), data=zero_padding)
            if pad_mode == "SAME_UPPER":
                x_tensor = tensor.concatenate((x_tensor, zero_padding), axis)
            else:
                x_tensor = tensor.concatenate((zero_padding, x_tensor), axis)
    return x_tensor.data


def handle_same_pad_bwd(dx, padding, pad_mode):
    """
    handle same padding mode backward
    Args:dx
        the backward tensor
    Args:padding
        the padding
    Returns: 
        tensor, the output
    """
    for axis, pad in zip((2, 3), padding):
        if pad % 2 != 0:
            axis_shape = list(dx.shape())[axis]
            if pad_mode == "SAME_UPPER":
                dx = singa.SliceOn(dx, 0, axis_shape - 1, axis)
            else:
                dx = singa.SliceOn(dx, 1, axis_shape, axis)
    return dx


def same_pad_shape_check(handle, pad_mode, x, padding):
    """
    check the shape is correct for same padding mode
    Args:handle
        the handle
    Args:pad_mode
        pad_mode
    Args:x
        input tensor
    Args:padding
        the padding
    """
    _kernel = [handle.kernel_h, handle.kernel_w]
    _stride = [handle.stride_h, handle.stride_w]
    output_shape = get_output_shape(pad_mode, x.shape()[2:], _kernel, _stride)
    _padding_correct = get_padding_shape(x.shape()[2:], _kernel, _stride,
                                         output_shape)
    _padding_correct = [x // 2 for x in _padding_correct]
    assert padding == _padding_correct, (
        'For a same mode, the given padding %s is wrong, the correct one should be %s.'
        % (padding, _padding_correct))


def re_new_handle(handle, x, is_pool=False):
    """
    re-new a handle by useing the new input tensor
    Args:handle
        the handle
    Args:x
        input tensor
    Returns: 
        handle, a new handle
    """
    kernel_size = [handle.kernel_h, handle.kernel_w]
    stride = [handle.stride_h, handle.stride_w]
    padding = [handle.pad_h, handle.pad_w]
    if is_pool:
        params = (x, kernel_size, stride, padding, handle.is_max_pooling)
    else:
        params = (x, kernel_size, stride, padding, handle.channels, handle.num_filters,
                handle.bias_term, handle.group)
    if (type(handle) == singa.ConvHandle or type(handle) == singa.PoolingHandle):
        handle = singa.PoolingHandle(*params) if is_pool else singa.ConvHandle(*params) 
    else:
        handle = singa.CudnnPoolingHandle(*params) if is_pool else singa.CudnnConvHandle(*params) 
    return handle


def get_padding_shape(input_spatial_shape, kernel_spatial_shape,
                      strides_spatial, output_spatial_shape):
    """
    return padding shape of conv2d or pooling,
    ! borrow from onnx
    Args:
        auto_pad: string
    Args:
        kernel_spatial_shape: list[int]
    Args:
        strides_spatial: list[int]
    Args:
        output_spatial_shape: list[int]
    Returns: 
        list[int]
    """
    pad_shape = [0] * len(input_spatial_shape)
    for i in range(len(input_spatial_shape)):
        pad_shape[i] = (output_spatial_shape[i] - 1) * strides_spatial[i] + \
            kernel_spatial_shape[i] - input_spatial_shape[i]
    return pad_shape


def get_output_shape(auto_pad, input_spatial_shape, kernel_spatial_shape,
                     strides_spatial):
    """
    return output shape of conv2d or pooling,
    ! borrow from onnx
    Args:
        auto_pad: string
    Args:
        kernel_spatial_shape: list[int]
    Args:
        strides_spatial: list[int]
    Args:
        output_spatial_shape: list[int]
    Returns: 
        list[int
    """
    out_shape = [0] * len(input_spatial_shape)
    if auto_pad in ('SAME_UPPER', 'SAME_LOWER'):
        for i in range(len(input_spatial_shape)):
            out_shape[i] = int(
                np.ceil(
                    float(input_spatial_shape[i]) / float(strides_spatial[i])))
    elif auto_pad == 'VALID':
        for i in range(len(input_spatial_shape)):
            out_shape[i] = int(
                np.ceil(
                    float(input_spatial_shape[i] -
                          (kernel_spatial_shape[i] - 1)) /
                    float(strides_spatial[i])))
    return out_shape


def force_unicode(s):
    """
    return string of a bytes
    ! borrow from onnx
    Args:
        s: string or bytes
    Returns: 
        string
    """
    try:
        return s.decode('utf-8')
    except AttributeError:
        return s
