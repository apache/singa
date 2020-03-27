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

from singa import opt
from singa import tensor
from singa import device
from singa import module
from singa import autograd

import numpy as np
import os
import sys
import gzip
import codecs


class CNN(module.Module):

    def __init__(self, sgd):
        super(CNN, self).__init__()

        self.conv1 = autograd.Conv2d(1, 20, 5, padding=0)
        self.conv2 = autograd.Conv2d(20, 50, 5, padding=0)
        self.linear1 = autograd.Linear(4 * 4 * 50, 500)
        self.linear2 = autograd.Linear(500, 10)
        self.pooling1 = autograd.MaxPool2d(2, 2, padding=0)
        self.pooling2 = autograd.MaxPool2d(2, 2, padding=0)

        self.sgd = sgd

    def forward(self, x):
        y = self.conv1(x)
        y = autograd.relu(y)
        y = self.pooling1(y)
        y = self.conv2(y)
        y = autograd.relu(y)
        y = self.pooling2(y)
        y = autograd.flatten(y)
        y = self.linear1(y)
        y = autograd.relu(y)
        y = self.linear2(y)
        return y

    def loss(self, x, ty):
        return autograd.softmax_cross_entropy(x, ty)

    def optim(self, loss):
        self.sgd.backward_and_update(loss)


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


# Function to all reduce NUMPY Accuracy and Loss from Multiple Devices
def reduce_variable(variable, dist_opt, reducer):
    reducer.copy_from_numpy(variable)
    dist_opt.all_reduce(reducer.data)
    dist_opt.wait()
    output = tensor.to_numpy(reducer)
    return output


# Function to sychronize SINGA TENSOR initial model parameters
def sychronize(tensor, dist_opt):
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


if __name__ == "__main__":
    sgd = opt.SGD(lr=0.005, momentum=0.9, weight_decay=1e-5)
    # sgd = opt.DistOpt(sgd)

    dev = device.create_cuda_gpu_on(0)

    # Prepare training and valadiation data
    batch_size = 64
    max_epoch = 10
    IMG_SIZE = 28
    num_classes = 10
    train_x, train_y, test_x, test_y = load_dataset()
    train_y = to_categorical(train_y, num_classes)
    test_y = to_categorical(test_y, num_classes)

    # Normalization
    train_x = train_x / 255
    test_x = test_x / 255

    tx = tensor.Tensor((batch_size, 1, IMG_SIZE, IMG_SIZE), dev, tensor.float32)
    ty = tensor.Tensor((batch_size, num_classes), dev, tensor.int32)
    num_train_batch = train_x.shape[0] // batch_size
    num_test_batch = test_x.shape[0] // batch_size
    idx = np.arange(train_x.shape[0], dtype=np.int32)

    model = CNN(sgd)
    model.on_device(dev)

    for epoch in range(max_epoch):
        np.random.shuffle(idx)

        # Training Phase
        autograd.training = True
        train_correct = np.zeros(shape=[1], dtype=np.float32)
        test_correct = np.zeros(shape=[1], dtype=np.float32)
        train_loss = np.zeros(shape=[1], dtype=np.float32)

        for b in range(num_train_batch):
            x = train_x[idx[b * batch_size:(b + 1) * batch_size]]
            x = augmentation(x, batch_size)
            y = train_y[idx[b * batch_size:(b + 1) * batch_size]]
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)

            out = model(tx)
            loss = model.loss(out, ty)
            model.optim(loss)

            train_correct += accuracy(tensor.to_numpy(out), y)
            train_loss += tensor.to_numpy(loss)[0]

        print('Training loss = %f, training accuracy = %f' %
              (train_loss, train_correct / (num_train_batch * batch_size)))
