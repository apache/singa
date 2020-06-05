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
import collections

from singa import tensor
from . import singa_wrap as singa

OrderedDict = collections.OrderedDict


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


def handle_odd_pad_fwd(x, odd_padding):
    """
    handle odd padding mode forward
    Args:
        x, the input tensor
        odd_padding, the odd_padding
    Returns: 
        tensor, the output
    """
    x_tensor = tensor.from_raw_tensor(x)
    # (axis, left padding if True else right padding)
    flags = [(2, True), (2, False), (3, True), (3, False)]
    for (axis, left), pad in zip(flags, odd_padding):
        if pad == 0:
            continue
        zeros_shape = list(x_tensor.data.shape())
        zeros_shape[axis] = pad
        zero_padding = np.zeros(zeros_shape).astype(np.float32)
        zero_padding = tensor.Tensor(device=x.device(), data=zero_padding)
        if left:
            x_tensor = tensor.concatenate((zero_padding, x_tensor), axis)
        else:
            x_tensor = tensor.concatenate((x_tensor, zero_padding), axis)
    return x_tensor.data


def handle_odd_pad_bwd(dx, odd_padding):
    """
    handle odd padding mode backward
    Args:
        dx, the backward tensor
        odd_padding, the odd_padding
    Returns: 
        tensor, the output
    """
    # (axis, left padding if True else right padding)
    flags = [(2, True), (2, False), (3, True), (3, False)]
    for (axis, left), pad in zip(flags, odd_padding):
        if pad == 0:
            continue
        axis_shape = list(dx.shape())[axis]
        if left:
            dx = singa.SliceOn(dx, pad, axis_shape, axis)
        else:
            dx = singa.SliceOn(dx, 0, axis_shape - pad, axis)
    return dx


def same_pad_shape_check(handle, pad_mode, x):
    """
    check the shape is correct for same padding mode
    Args:
        handle, the handle
        pad_mode, pad_mode
        x: input tensor
    Returns: 
        tuple, the correct padding(before divide 2)
    """
    _kernel = [handle.kernel_h, handle.kernel_w]
    _stride = [handle.stride_h, handle.stride_w]
    _padding = [handle.pad_h, handle.pad_w]
    _padding_correct = get_padding_shape(pad_mode,
                                         x.shape()[2:], _kernel, _stride)
    _padding_crop, _ = [x // 2 for x in _padding_correct]
    assert _padding == _padding_crop, (
        'For a same mode, the given padding %s is wrong, the correct one should be %s.'
        % (_padding, _padding_crop))
    return _padding_correct


def re_new_handle(handle, x, is_pool=False):
    """
    re-new a handle by useing the new input tensor
    Args:
        handle, the handle
        x, input tensor
    Returns: 
        handle, a new handle
    """
    kernel_size = [handle.kernel_h, handle.kernel_w]
    stride = [handle.stride_h, handle.stride_w]
    padding = [handle.pad_h, handle.pad_w]
    if is_pool:
        params = (x, kernel_size, stride, padding, handle.is_max_pooling)
    else:
        params = (x, kernel_size, stride, padding, handle.channels,
                  handle.num_filters, handle.bias_term, handle.group)
    if (type(handle) == singa.ConvHandle or
            type(handle) == singa.PoolingHandle):
        handle = singa.PoolingHandle(*params) if is_pool else singa.ConvHandle(
            *params)
    else:
        handle = singa.CudnnPoolingHandle(
            *params) if is_pool else singa.CudnnConvHandle(*params)
    return handle


def get_padding_shape(pad_mode, input_spatial_shape, kernel_spatial_shape,
                      strides_spatial):
    """
    return padding shape of conv2d or pooling,
    Args:
        pad_mode: string
        kernel_spatial_shape: list[int]
        strides_spatial: list[int]
    Returns: 
        list[int]
    """
    output_spatial_shape = get_output_shape(pad_mode, input_spatial_shape,
                                            kernel_spatial_shape,
                                            strides_spatial)
    pad_shape = [0] * len(input_spatial_shape) * 2  # 2 means left and right
    # the odd paddding is the value that cannot be handled by the tuple padding (w, h) mode
    # so we need to firstly handle the input, then use the nomal padding method.
    odd_padd_shape = [0] * len(input_spatial_shape) * 2
    for i in range(len(input_spatial_shape)):
        whole_pad = (output_spatial_shape[i] - 1) * strides_spatial[i] + \
            kernel_spatial_shape[i] - input_spatial_shape[i]
        pad_shape[2 * i] = pad_shape[2 * i + 1] = whole_pad // 2
        if whole_pad % 2 != 0:
            if pad_mode == "SAME_UPPER":
                odd_padd_shape[2 * i + 1] += 1
            else:
                odd_padd_shape[2 * i] += 1
    return pad_shape, odd_padd_shape


def get_output_shape(auto_pad, input_spatial_shape, kernel_spatial_shape,
                     strides_spatial):
    """
    return output shape of conv2d or pooling,
    ! borrow from onnx
    Args:
        auto_pad: string
        input_spatial_shape: list[int]
        kernel_spatial_shape: list[int]
        strides_spatial: list[int]
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


def post_order_recursive(root, root_t):
    """
    return a list by the topological ordering (postorder of Depth-first search)
    Args:
        root: singa operator
        root_t: tensor
    Returns: 
        deque[int]
    """

    def recursive(root, yid, root_t, nodes, weights, inputs):
        if root:
            # srcop: operator for a input of root
            # yid: id(output of this operator)
            # y: output of this operator
            for srcop, yid, y, _ in root.src:
                recursive(srcop, yid, y, nodes, weights, inputs)

            if type(root).__name__ == 'Dummy':
                if root_t != None:
                    # constant within a node: weight
                    weights[root.name] = root_t
                else:
                    # constant outside a node: input
                    inputs[root.name] = root_t
            else:
                nodes[root.name] = root

    nodes = OrderedDict()
    weights = OrderedDict()
    inputs = OrderedDict()

    recursive(root, None, root_t, nodes, weights, inputs)
    return nodes, weights, inputs
