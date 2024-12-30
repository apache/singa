#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

try:
    import pickle
except ImportError:
    import cPickle as pickle

import os
import sys
import random
import numpy as np
from PIL import Image


# need to save to specific local directories
def load_data(dir_path="/tmp/diaret", resize_size=(128, 128)):
    dir_path = check_dataset_exist(dirpath=dir_path)
    image_sets = {label: load_image_path(os.listdir(os.path.join(dir_path, label)))
            for label in os.listdir(dir_path)}
    images, labels = [], []
    for label in os.listdir(dir_path):
        image_names = load_image_path(os.listdir(os.path.join(dir_path, label)))
        label_images = [np.array(Image.open(os.path.join(dir_path, label, img_name))\
                .resize(resize_size).convert("RGB")).transpose(2, 0, 1)
                for img_name in image_names]
        images.extend(label_images)
        labels.extend([int(label)] * len(label_images))

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return images, labels


def load_image_path(image_pths):
    allowed_image_format = ['png', 'jpg', 'jpeg']
    return list(filter(lambda pth: pth.rsplit('.')[-1].lower() in allowed_image_format,
        image_pths))


def check_dataset_exist(dirpath):
    if not os.path.exists(dirpath):
        print(
            'Please download the Diabetic Retinopathy dataset first'
        )
        sys.exit(0)
    return dirpath


def normalize(train_x, val_x):
    mean = [0.5339, 0.4180, 0.4460]  # mean for dataset
    std = [0.3329, 0.2637, 0.2761]  # std for dataset
    train_x /= 255
    val_x /= 255
    for ch in range(0, 2):
        train_x[:, ch, :, :] -= mean[ch]
        train_x[:, ch, :, :] /= std[ch]
        val_x[:, ch, :, :] -= mean[ch]
        val_x[:, ch, :, :] /= std[ch]
    return train_x, val_x


def train_test_split(x, y, val_ratio=0.2):
    indices = list(range(len(x)))
    val_indices = list(random.sample(indices, int(val_ratio*len(x))))
    train_indices = list(set(indices) - set(val_indices))
    return x[train_indices], y[train_indices], x[val_indices], y[val_indices]


def load(dir_path):
    x, y = load_data(dir_path=dir_path)
    train_x, train_y, val_x, val_y = train_test_split(x, y)
    train_x, val_x = normalize(train_x, val_x)
    return train_x, train_y, val_x, val_y
