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

from singa import singa_wrap as singa
from singa import autograd
from singa import tensor
from singa import device
from singa import opt
from PIL import Image
import numpy as np
import os
import sys
import time


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


def load_train_data(dir_path='cifar-10-batches-py', num_batches=5):
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


def load_test_data(dir_path='cifar-10-batches-py'):
    images, labels = load_dataset(check_dataset_exist(dir_path + "/test_batch"))
    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32)


def check_dataset_exist(dirpath):
    if not os.path.exists(dirpath):
        print(
            'Please download the cifar10 dataset using download_data.py (e.g. python ~/singa/examples/cifar10/download_data.py py)'
        )
        sys.exit(0)
    return dirpath


def normalize_for_resnet(train_x, test_x):
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]
    train_x /= 255
    test_x /= 255
    for ch in range(0, 2):
        train_x[:, ch, :, :] -= mean[ch]
        train_x[:, ch, :, :] /= std[ch]
        test_x[:, ch, :, :] -= mean[ch]
        test_x[:, ch, :, :] /= std[ch]
    return train_x, test_x


def resize_dataset(x, IMG_SIZE):
    num_data = x.shape[0]
    dim = x.shape[1]
    X = np.zeros(shape=(num_data, dim, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for n in range(0, num_data):
        for d in range(0, dim):
            X[n, d, :, :] = np.array(Image.fromarray(x[n, d, :, :]).resize(
                (IMG_SIZE, IMG_SIZE), Image.BILINEAR),
                                     dtype=np.float32)
    return X


def augmentation(x, batch_size):
    xpad = np.pad(x, [[0, 0], [0, 0], [4, 4], [4, 4]], 'symmetric')
    for data_num in range(0, batch_size):
        offset = np.random.randint(8, size=2)
        x[data_num, :, :, :] = xpad[data_num, :, offset[0]:offset[0] + 32,
                                    offset[1]:offset[1] + 32]
        if_flip = np.random.randint(2)
        if (if_flip):
            x[data_num, :, :, :] = x[data_num, :, :, ::-1]
    return x


def accuracy(pred, target):
    y = np.argmax(pred, axis=1)
    t = np.argmax(target, axis=1)
    a = y == t
    return np.array(a, "int").sum()


def to_categorical(y, num_classes):
    y = np.array(y, dtype="int")
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    for i in range(0, n):
        categorical[i, y[i]] = 1
        categorical = categorical.astype(np.float32)
    return categorical


# Function to all reduce NUMPY accuracy and loss from multiple devices
def reduce_variable(variable, dist_opt, reducer):
    reducer.copy_from_numpy(variable)
    dist_opt.all_reduce(reducer.data)
    dist_opt.wait()
    output = tensor.to_numpy(reducer)
    return output


# Function to synchronize SINGA TENSOR initial model parameters
def synchronize(tensor, dist_opt):
    dist_opt.all_reduce(tensor.data)
    dist_opt.wait()
    tensor /= dist_opt.world_size


# Data partition
def data_partition(dataset_x, dataset_y, global_rank, world_size):
    data_per_rank = dataset_x.shape[0] // world_size
    idx_start = global_rank * data_per_rank
    idx_end = (global_rank + 1) * data_per_rank
    return dataset_x[idx_start:idx_end], dataset_y[idx_start:idx_end]
