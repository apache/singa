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

from singa import singa_wrap as singa
from singa import autograd
from singa import layer
from singa import tensor
from singa import device
from singa import opt
import numpy as np
import os
import sys
import gzip
import codecs
import time


class CNN:

    def __init__(self):
        self.conv1 = layer.Conv2d(1, 20, 5, padding=0)
        self.conv2 = layer.Conv2d(20, 50, 5, padding=0)
        self.linear1 = layer.Linear(4 * 4 * 50, 500)
        self.linear2 = layer.Linear(500, 10)
        self.pooling1 = layer.MaxPool2d(2, 2, padding=0)
        self.pooling2 = layer.MaxPool2d(2, 2, padding=0)
        self.relu1 = layer.ReLU()
        self.relu2 = layer.ReLU()
        self.relu3 = layer.ReLU()
        self.flatten = layer.Flatten()

    def forward(self, x):
        y = self.conv1(x)
        y = self.relu1(y)
        y = self.pooling1(y)
        y = self.conv2(y)
        y = self.relu2(y)
        y = self.pooling2(y)
        y = self.flatten(y)
        y = self.linear1(y)
        y = self.relu3(y)
        y = self.linear2(y)
        return y


def check_dataset_exist(dirpath):
    if not os.path.exists(dirpath):
        print(
            'The MNIST dataset does not exist. Please download the mnist dataset using download_mnist.py (e.g. python3 download_mnist.py)'
        )
        sys.exit(0)
    return dirpath


def load_dataset():
    train_x_path = '/tmp/train-images-idx3-ubyte.gz'
    train_y_path = '/tmp/train-labels-idx1-ubyte.gz'
    valid_x_path = '/tmp/t10k-images-idx3-ubyte.gz'
    valid_y_path = '/tmp/t10k-labels-idx1-ubyte.gz'

    train_x = read_image_file(check_dataset_exist(train_x_path)).astype(
        np.float32)
    train_y = read_label_file(check_dataset_exist(train_y_path)).astype(
        np.float32)
    valid_x = read_image_file(check_dataset_exist(valid_x_path)).astype(
        np.float32)
    valid_y = read_label_file(check_dataset_exist(valid_y_path)).astype(
        np.float32)
    return train_x, train_y, valid_x, valid_y


def read_label_file(path):
    with gzip.open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8).reshape((length))
        return parsed


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_image_file(path):
    with gzip.open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16).reshape(
            (length, 1, num_rows, num_cols))
        return parsed


def to_categorical(y, num_classes):
    y = np.array(y, dtype="int")
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    categorical = categorical.astype(np.float32)
    return categorical


def accuracy(pred, target):
    y = np.argmax(pred, axis=1)
    t = np.argmax(target, axis=1)
    a = y == t
    return np.array(a, "int").sum()


# Function to all reduce NUMPY accuracy and loss from multiple devices
def reduce_variable(variable, dist_opt, reducer):
    reducer.copy_from_numpy(variable)
    dist_opt.all_reduce(reducer.data)
    dist_opt.wait()
    output = tensor.to_numpy(reducer)
    return output


# Function to sychronize SINGA TENSOR initial model parameters
def synchronize(tensor, dist_opt):
    dist_opt.all_reduce(tensor.data)
    dist_opt.wait()
    tensor /= dist_opt.world_size


# Data augmentation
def augmentation(x, batch_size):
    xpad = np.pad(x, [[0, 0], [0, 0], [4, 4], [4, 4]], 'symmetric')
    for data_num in range(0, batch_size):
        offset = np.random.randint(8, size=2)
        x[data_num, :, :, :] = xpad[data_num, :, offset[0]:offset[0] + 28,
                                    offset[1]:offset[1] + 28]
        if_flip = np.random.randint(2)
        if (if_flip):
            x[data_num, :, :, :] = x[data_num, :, :, ::-1]
    return x


# Data partition
def data_partition(dataset_x, dataset_y, global_rank, world_size):
    data_per_rank = dataset_x.shape[0] // world_size
    idx_start = global_rank * data_per_rank
    idx_end = (global_rank + 1) * data_per_rank
    return dataset_x[idx_start:idx_end], dataset_y[idx_start:idx_end]
