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

try:
    import pickle
except ImportError:
    import cPickle as pickle

import numpy as np
import os
import sys


def load_dataset(filepath):
    with open(filepath, 'rb') as fd:
        try:
            cifar10 = pickle.load(fd, encoding='latin1')
        except TypeError:
            cifar10 = pickle.load(fd)
    image = cifar10['data'].astype(dtype=np.uint8)
    image = image.reshape((-1, 3, 32, 32))
    label = np.asarray(cifar10['labels'], dtype=np.uint8)
    label = label.reshape(label.size, 1)
    return image, label


def load_train_data(dir_path='/tmp/cifar-10-batches-py', num_batches=5):
    labels = []
    batchsize = 10000
    images = np.empty((num_batches * batchsize, 3, 32, 32), dtype=np.uint8)
    for did in range(1, num_batches + 1):
        fname_train_data = dir_path + "/data_batch_{}".format(did)
        image, label = load_dataset(check_dataset_exist(fname_train_data))
        images[(did - 1) * batchsize:did * batchsize] = image
        labels.extend(label)
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return images, labels


def load_test_data(dir_path='/tmp/cifar-10-batches-py'):
    images, labels = load_dataset(check_dataset_exist(dir_path + "/test_batch"))
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


def check_dataset_exist(dirpath):
    if not os.path.exists(dirpath):
        print(
            'Please download the cifar10 dataset using python data/download_cifar10.py'
        )
        sys.exit(0)
    return dirpath


def normalize(train_x, val_x):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    train_x /= 255
    val_x /= 255
    for ch in range(0, 2):
        train_x[:, ch, :, :] -= mean[ch]
        train_x[:, ch, :, :] /= std[ch]
        val_x[:, ch, :, :] -= mean[ch]
        val_x[:, ch, :, :] /= std[ch]
    return train_x, val_x

def load():
    train_x, train_y = load_train_data()
    val_x, val_y = load_test_data()
    train_x, val_x = normalize(train_x, val_x)
    train_y = train_y.flatten()
    val_y = val_y.flatten()
    return train_x, train_y, val_x, val_y
