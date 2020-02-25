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


def handle_same_pad_fwd(y, pad_mode):
    """
    handle same padding mode forward
    Args:dy
        the forward tensor
    Returns: 
        tensor, the output
    """
    y_shape = y.shape()
    y = tensor.from_raw_tensor(y)
    if y_shape[2] == 1:
        label_1, label_2 = 0, 1
    else:
        label_1, label_2 = 1, 1
    if pad_mode == "SAME_UPPER":
        y = y[:, :, label_1:, label_2:]
    elif pad_mode == "SAME_LOWER":
        y = y[:, :, :-label_1, :-label_2]
    return y.data


def handle_same_pad_bwd(dy, pad_mode):
    """
    handle same padding mode backward
    Args:dy
        the backward tensor
    Returns: 
        tensor, the output
    """
    dy_shape = dy.shape()
    # one column zeros at last axis
    padding_1 = np.zeros([*dy_shape[:3], 1]).astype(np.float32)
    padding_1 = tensor.Tensor(device=dy.device(), data=padding_1)
    dy_tensor = tensor.from_raw_tensor(dy)
    if pad_mode == "SAME_UPPER":
        concat_left, concat_right = padding_1, dy_tensor
    else:
        concat_left, concat_right = dy_tensor, padding_1
    dy_tensor = tensor.concatenate((concat_left, concat_right), 3)
    if dy_shape[2] != 1:  # if not 1d
        # one row zeros at last second axis
        padding_2 = np.zeros([*dy_shape[:2], 1,
                              dy_shape[-1] + 1]).astype(np.float32)
        padding_2 = tensor.Tensor(device=dy.device(), data=padding_2)
        if pad_mode == "SAME_UPPER":
            concat_left, concat_right = padding_2, dy_tensor
        else:
            concat_left, concat_right = dy_tensor, padding_2
        dy_tensor = tensor.concatenate((dy_tensor, padding_2), 2)
    return dy_tensor.data


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
        # once need padding along one direction
        if pad_shape[i] % 2 != 0:
            pad_shape[i] = int(math.ceil(pad_shape[i] / 2)) * strides_spatial[i]
        else:
            pad_shape[i] = pad_shape[i] // 2
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
