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

import os
import json
from glob import glob
import numpy as np
from PIL import Image


class Compose(object):
    """Compose several transforms together.

    Args:
        transforms: list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.ToTensor(),
        >>>     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        >>> ])

    """

    def __init__(self, transforms):
        self.transforms = transforms

    def forward(self, img):
        """
        Args:
            img (PIL Image or numpy array): Image to be processed.

        Returns:
            PIL Image or numpy array: Processed image.
        """
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


class ToTensor(object):
    """Convert a ``PIL Image`` to ``numpy.ndarray``.

    Converts a PIL Image (H x W x C) in the range [0, 255] to a ``numpy.array`` of shape
    (C x H x W) in the range [0.0, 1.0]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1).

    In the other cases, tensors are returned without scaling.

    .. note::
        Because the input image is scaled to [0.0, 1.0], this transformation should not be used when
        transforming target image masks.
    """

    def forward(self, pic):
        """
        Args:
            pic (PIL Image): Image to be converted to array.

        Returns:
            Array: Converted image.
        """
        if not isinstance(pic, Image.Image):
            raise TypeError('pic should be PIL Image. Got {}'.format(type(pic)))

        # Handle PIL Image
        mode_to_nptype = {'I': np.int32, 'I;16': np.int16, 'F': np.float32}
        img = np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True)

        if pic.mode == '1':
            img = 255 * img

        # Put it from HWC to CHW format
        img = np.transpose(img, (2, 0, 1))

        if img.dtype == np.uint8:
            return np.array(np.float32(img) / 255.0, dtype=np.float)
        else:
            return np.float(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Normalize(object):
    """Normalize a ``numpy.array`` image with mean and standard deviation.

    This transform does not support PIL Image.
    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
    channels, this transform will normalize each channel of the input
    ``numpy.array`` i.e.,
    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutate the input array.

    Args:
        mean (Sequence): Sequence of means for each channel.
        std (Sequence): Sequence of standard deviations for each channel.
        inplace(bool, optional): Bool to make this operation in-place.

    """

    def __init__(self, mean, std, inplace=False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, img: np.ndarray):
        """
        Args:
            img (Numpy ndarray): Array image to be normalized.

        Returns:
            d_res (Numpy ndarray): Normalized Tensor image.
        """
        if not isinstance(img, np.ndarray):
            raise TypeError('Input img should be a numpy array. Got {}.'.format(type(img)))

        if not img.dtype == np.float:
            raise TypeError('Input array should be a float array. Got {}.'.format(img.dtype))

        if img.ndim < 3:
            raise ValueError('Expected array to be an array image of size (..., C, H, W). Got img.shape = '
                             '{}.'.format(img.shape))

        if not self.inplace:
            img = img.copy()

        dtype = img.dtype
        mean = np.array(self.mean, dtype=dtype)
        std = np.array(self.std, dtype=dtype)
        if (std == 0).any():
            raise ValueError('std evaluated to zero after conversion to {}, leading to division by zero.'.format(dtype))
        s_res = np.subtract(img, mean[:, None, None])
        d_res = np.divide(s_res, std[:, None, None])

        return d_res

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class ClassDataset(object):
    """Fetch data from file and generate batches.

    Load data from folder as PIL.Images and convert them into batch array.

    Args:
        img_folder (Str): Folder path of the training/validation images.
        transforms (Transform):  Preprocess transforms.
    """

    def __init__(self, img_folder, transforms):
        super(ClassDataset, self).__init__()

        self.img_list = list()
        self.transforms = transforms

        classes = os.listdir(img_folder)
        for i in classes:
            images = glob(os.path.join(img_folder, i, "*"))
            for img in images:
                self.img_list.append((img, i))

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, index: int):
        img_path, label_str = self.img_list[index]
        img = Image.open(img_path)
        img = self.transforms.forward(img)
        label = np.array(label_str, dtype=np.int32)

        return img, label

    def batchgenerator(self, indexes, batch_size, data_size):
        """Generate batch arrays from transformed image list.

        Args:
            indexes (Sequence): current batch indexes list, e.g. [n, n + 1, ..., n + batch_size]
            batch_size (int):
            data_size (Tuple): input image size of shape (C, H, W)

        Return:
            batch_x (Numpy ndarray): batch array of input images (B, C, H, W)
            batch_y (Numpy ndarray): batch array of ground truth lables (B,)
        """
        batch_x = np.zeros((batch_size,) + data_size)
        batch_y = np.zeros((batch_size,) + (1,), dtype=np.int32)
        for idx, i in enumerate(indexes):
            sample_x, sample_y = self.__getitem__(i)
            batch_x[idx, :, :, :] = sample_x
            batch_y[idx, :] = sample_y

        return batch_x, batch_y


def load(dir_path="tmp/bloodmnist"):
    # Dataset loading
    train_path = os.path.join(dir_path, "train")
    val_path = os.path.join(dir_path, "val")
    cfg_path = os.path.join(dir_path, "param.json")

    with open(cfg_path, 'r') as load_f:
        num_class = json.load(load_f)["num_classes"]

    # Define pre-processing methods (transforms)
    transforms = Compose([
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_dataset = ClassDataset(train_path, transforms)
    val_dataset = ClassDataset(val_path, transforms)
    return train_dataset, val_dataset, num_class
