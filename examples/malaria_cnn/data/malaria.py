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

import numpy as np
import os
import sys
from PIL import Image


# need to save to specific local directories
def load_train_data(dir_path="/tmp/malaria", resize_size=(128, 128)):
    dir_path = check_dataset_exist(dirpath=dir_path)
    path_train_label_1 = os.path.join(dir_path, "training_set/Parasitized")
    path_train_label_0 = os.path.join(dir_path, "training_set/Uninfected")
    train_label_1 = load_image_path(os.listdir(path_train_label_1))
    train_label_0 = load_image_path(os.listdir(path_train_label_0))
    labels = []
    Images = np.empty((len(train_label_1) + len(train_label_0), 3,
                       resize_size[0], resize_size[1]),
                      dtype=np.uint8)
    for i in range(len(train_label_0)):
        image_path = os.path.join(path_train_label_0, train_label_0[i])
        temp_image = np.array(
            Image.open(image_path).resize(resize_size).convert(
                "RGB")).transpose(2, 0, 1)
        Images[i] = temp_image
        labels.append(0)
    for i in range(len(train_label_1)):
        image_path = os.path.join(path_train_label_1, train_label_1[i])
        temp_image = np.array(
            Image.open(image_path).resize(resize_size).convert(
                "RGB")).transpose(2, 0, 1)
        Images[i + len(train_label_0)] = temp_image
        labels.append(1)

    Images = np.array(Images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return Images, labels


# need to save to specific local directories
def load_test_data(dir_path='/tmp/malaria', resize_size=(128, 128)):
    dir_path = check_dataset_exist(dirpath=dir_path)
    path_test_label_1 = os.path.join(dir_path, "testing_set/Parasitized")
    path_test_label_0 = os.path.join(dir_path, "testing_set/Uninfected")
    test_label_1 = load_image_path(os.listdir(path_test_label_1))
    test_label_0 = load_image_path(os.listdir(path_test_label_0))
    labels = []
    Images = np.empty((len(test_label_1) + len(test_label_0), 3, resize_size[0],
                       resize_size[1]),
                      dtype=np.uint8)
    for i in range(len(test_label_0)):
        image_path = os.path.join(path_test_label_0, test_label_0[i])
        temp_image = np.array(
            Image.open(image_path).resize(resize_size).convert(
                "RGB")).transpose(2, 0, 1)
        Images[i] = temp_image
        labels.append(0)
    for i in range(len(test_label_1)):
        image_path = os.path.join(path_test_label_1, test_label_1[i])
        temp_image = np.array(
            Image.open(image_path).resize(resize_size).convert(
                "RGB")).transpose(2, 0, 1)
        Images[i + len(test_label_0)] = temp_image
        labels.append(1)

    Images = np.array(Images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return Images, labels


def load_image_path(list):
    new_list = []
    for image_path in list:
        if (image_path.endswith(".png") or image_path.endswith(".jpg")):
            new_list.append(image_path)
    return new_list


def check_dataset_exist(dirpath):
    if not os.path.exists(dirpath):
        print('Please download the malaria dataset first')
        sys.exit(0)
    return dirpath


def normalize(train_x, val_x):
    mean = [0.5339, 0.4180, 0.4460]  # mean for malaria dataset
    std = [0.3329, 0.2637, 0.2761]  # std for malaria dataset
    train_x /= 255
    val_x /= 255
    for ch in range(0, 2):
        train_x[:, ch, :, :] -= mean[ch]
        train_x[:, ch, :, :] /= std[ch]
        val_x[:, ch, :, :] -= mean[ch]
        val_x[:, ch, :, :] /= std[ch]
    return train_x, val_x


def load(dir_path):
    train_x, train_y = load_train_data(dir_path=dir_path)
    val_x, val_y = load_test_data(dir_path=dir_path)
    train_x, val_x = normalize(train_x, val_x)
    train_y = train_y.flatten()
    val_y = val_y.flatten()
    return train_x, train_y, val_x, val_y
