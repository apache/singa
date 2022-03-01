#
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
#

from textwrap import fill
from turtle import forward
import numpy as np
from PIL import Image, ImageOps
from collections.abc import Sequence
import numbers


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def forward(self, img):
        for t in self.transforms:
            img = t.forward(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


class ToTensor:
    def forward(self, pic):
        """
        Args:
            pic (PIL Image): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        if not isinstance(pic, Image.Image):
           raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        # handle PIL Image
        mode_to_nptype = {'I': np.int32, 'I;16': np.int16, 'F': np.float32}
        img = np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True)

        if pic.mode == '1':
            img = 255 * img

        # put it from HWC to CHW format
        img = np.transpose(img, (2, 0, 1))

        if img.dtype == np.uint8:
            return np.array(np.float32(img)/255.0, dtype=np.float)
        else:
            return np.float(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize:
    """Normalize a tensor image with mean and standard deviation.
    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``torch.*Tensor`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, img: np.ndarray) -> np.ndarray:
        """
        Args:
            tensor (Tensor): Tensor image to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if not isinstance(img, np.ndarray):
            raise TypeError('Input img should be a numpy array. Got {}.'.format(type(img)))

        if not img.dtype == np.float:
            raise TypeError('Input tensor should be a float tensor. Got {}.'.format(img.dtype))

        if img.ndim < 3:
            raise ValueError('Expected tensor to be a tensor image of size (..., C, H, W). Got img.shape = '
                            '{}.'.format(img.shape))

        if not self.inplace:
            img = img.copy()

        dtype = img.dtype
        mean = np.array(self.mean, dtype=dtype)
        std = np.array(self.std, dtype=dtype)
        if (std == 0).any():
            raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
        # if mean.ndim == 1:
        #     mean = mean.view(-1, 1, 1)
        # if std.ndim == 1:
        #     std = std.view(-1, 1, 1)
        # tensor.sub_(mean).div_(std)
        s_res = np.subtract(img, mean[:, None, None])
        d_res = np.divide(s_res, std[:, None, None])

        return d_res


    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Resize:
    def __init__(self, size, interpolation="BILINEAR"):
        if not isinstance(size, (int, Sequence)):
            raise TypeError("Size should be int or sequence. Got {}".format(type(size)))
        if isinstance(size, Sequence) and len(size) not in (1, 2):
            raise ValueError("If size is a sequence, it should have 1 or 2 values")
        self.size = size

        # Backward compatibility with integer value
        interpolation_method = {
            'BILINEAR': Image.BILINEAR,
            'CUBIC': Image.BICUBIC,
            'NEAREST': Image.NEAREST,
            'BOX': Image.BOX,
            'HAMMING': Image.HAMMING,
            'LANCZOS': Image.LANCZOS
        }

        self.interpolation = interpolation_method[interpolation]
    def forward(self, image):
        return image.resize(self.size, resample=self.interpolation, box=None, reducing_gap=None)


    def __repr__(self):
        interpolate_str = self.interpolation
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(
            self.size, interpolate_str)


class Pad:
    def __init__(self, padding, fill=0, padding_mode="constant"):
        if not isinstance(padding, (numbers.Number, tuple, list)):
            raise TypeError("Got inappropriate padding arg")

        if not isinstance(fill, (numbers.Number, str, tuple)):
            raise TypeError("Got inappropriate fill arg")

        if padding_mode not in ["constant", "edge", "reflect", "symmetric"]:
            raise ValueError("Padding mode should be either constant, edge, reflect or symmetric")

        if isinstance(padding, Sequence) and len(padding) not in [1, 2, 4]:
            raise ValueError("Padding must be an int or a 1, 2, or 4 element tuple, not a " +
                             "{} element tuple".format(len(padding)))

        self.padding = padding
        self.fill = fill
    
    def forward(self, img):
        return ImageOps.expand(img, border=self.padding, fill=self.fill)
    
    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}'.\
            format(self.padding, self.fill)


class SquareResizePad:
    def __init__(self, target_length, interpolation_strategy="BILINEAR", pad_value=255):
        self.target_length = target_length
        self.interpolation_strategy = interpolation_strategy
        self.pad_value = pad_value

        # Pad.__init__(self, padding=(0, 0, 0, 0),
        #              fill=self.pad_value, padding_mode="constant")
        # Resize.__init__(self, size=(512, 512),
        #                 interpolation=self.interpolation_strategy)

    def forward(self, img):
        w, h = img.size
        if w > h:
            target_size = (
                int(np.round(self.target_length * (h / w))), self.target_length)
            # self.size = (int(np.round(self.target_length * (h / w))), self.target_length)
            # img = Resize.forward(self, img)
            resize = Resize(size=target_size, interpolation=self.interpolation_strategy)
            img = resize.forward(img)

            total_pad = target_size[1] - target_size[0]
            # total_pad = self.size[1] - self.size[0]
            half_pad = total_pad // 2
            padding = (0, half_pad, 0, total_pad - half_pad)
            # self.padding = (0, half_pad, 0, total_pad - half_pad)
            pad = Pad(padding=padding, fill=self.pad_value)
            return pad.forward(self, img)
        else:
            target_size = (self.target_length, 
                int(np.round(self.target_length * (w / h))))
            # self.size = (int(np.round(self.target_length * (h / w))), self.target_length)
            # img = Resize.forward(self, img)
            resize = Resize(size=target_size, interpolation=self.interpolation_strategy)
            img = resize.forward(img)

            total_pad = target_size[0] - target_size[1]
            # total_pad = self.size[0] - self.size[1]
            half_pad = total_pad // 2
            padding = (half_pad, 0, total_pad - half_pad, 0)
            # self.padding = (half_pad, 0, total_pad - half_pad, 0)
            pad = Pad(padding=padding, fill=self.pad_value)
            return pad.forward(img)

