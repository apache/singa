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

"""
1. Export singa model to onnx
2. Load onnx model and run it via singa backend
"""

import numpy as np
import os

from singa import device
from singa import tensor
from singa import autograd
from singa import opt
from singa import sonnx


def load_data(path):
    f = np.load(path)
    x_train, y_train = f["x_train"], f["y_train"]
    x_test, y_test = f["x_test"], f["y_test"]
    f.close()
    return (x_train, y_train), (x_test, y_test)


def to_categorical(y, num_classes):
    """
    Converts a class vector (integers) to binary class matrix.
    Args
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    Return
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype="int")
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    categorical = categorical.astype(np.float32)
    return categorical


def preprocess(data):
    data = data.astype(np.float32)
    data /= 255
    data = np.expand_dims(data, axis=1)
    return data


def common(use_cpu):
    file_path = "mnist.npz"
    assert os.path.exists(
        file_path
    ), "Pls download the MNIST dataset from https://s3.amazonaws.com/img-datasets/mnist.npz"
    if use_cpu:
        print("Using CPU")
        dev = device.get_default_device()
    else:
        print("Using GPU")
        dev = device.create_cuda_gpu()

    train, test = load_data(file_path)
    print(train[0].shape)
    x_train = preprocess(train[0])
    y_train = to_categorical(train[1], 10)

    x_test = preprocess(test[0])
    y_test = to_categorical(test[1], 10)
    print("the shape of training data is", x_train.shape)
    print("the shape of training label is", y_train.shape)
    print("the shape of testing data is", x_test.shape)
    print("the shape of testing label is", y_test.shape)
    return (x_train, y_train), (x_test, y_test), dev


def accuracy(pred, target):
    y = np.argmax(pred, axis=1)
    t = np.argmax(target, axis=1)
    a = y == t
    return np.array(a, "int").sum() / float(len(t))


def singa_to_onnx(epochs, use_cpu=False, batchsize=32):
    sgd = opt.SGD(lr=0.1)

    # operations initialization
    conv1 = autograd.Conv2d(1, 8, 3, 2, padding=1) # 28 - 14
    conv2 = autograd.Conv2d(8, 4, 3, 2, padding=1) # 14 - 7
    pooling = autograd.MaxPool2d(3, 2, padding=1) # 7 - 4
    linear = autograd.Linear(64, 10)

    def forward(x, t):
        y = conv1(x)
        y = autograd.relu(y)
        y = conv2(y)
        y = autograd.relu(y)
        y = pooling(y)
        y = autograd.flatten(y)
        y = linear(y)
        loss = autograd.softmax_cross_entropy(y, t)
        return loss, y

    autograd.training = True
    (x_train, y_train), (x_test, y_test), dev = common(use_cpu)

    niter = 1 # x_train.shape[0] // batchsize
    for epoch in range(epochs):
        accuracy_rate = 0.0
        loss_rate = 0.0
        for i in range(niter):
            inputs = tensor.Tensor(
                device=dev,
                data=x_train[i * batchsize : (i + 1) * batchsize],
                stores_grad=False,
                name="input",
            )
            targets = tensor.Tensor(
                device=dev,
                data=y_train[i * batchsize : (i + 1) * batchsize],
                requires_grad=False,
                stores_grad=False,
                name="target",
            )
            loss, y = forward(inputs, targets)
            accuracy_rate += accuracy(
                tensor.to_numpy(y), y_train[i * batchsize : (i + 1) * batchsize]
            )
            loss_rate += tensor.to_numpy(loss)[0]
            for p, gp in autograd.backward(loss):
                sgd.update(p, gp)
        print( "accuracy is {}, loss is {}".format( accuracy_rate / niter, loss_rate / niter))
    model = sonnx.to_onnx_model([inputs], [y])
    sonnx.save(model, "cnn.onnx")


def onnx_to_singa(epochs, use_cpu=False, batchsize=32):
    (x_train, y_train), (x_test, y_test), dev = common(use_cpu)
    model = sonnx.load("cnn.onnx")
    backend = sonnx.prepare(model, dev)
    autograd.training = True
    sgd = opt.SGD(lr=0.01)
    niter = x_train.shape[0] // batchsize
    for epoch in range(epochs):
        accuracy_rate = 0.0
        loss_rate = 0.0
        for i in range(niter):
            inputs = tensor.Tensor(
                device=dev,
                data=x_train[i * batchsize : (i + 1) * batchsize],
                stores_grad=False,
                name="input",
            )
            targets = tensor.Tensor(
                device=dev,
                data=y_train[i * batchsize : (i + 1) * batchsize],
                requires_grad=False,
                stores_grad=False,
                name="target",
            )
            y = backend.run([inputs])[0]
            loss = autograd.softmax_cross_entropy(y, targets)

            accuracy_rate += accuracy(
                tensor.to_numpy(y), y_train[i * batchsize : (i + 1) * batchsize]
            )
            loss_rate += tensor.to_numpy(loss)[0]

            for p, gp in autograd.backward(loss):
                sgd.update(p, gp)

        print("accuracy is {}, loss is {}".format(accuracy_rate / niter, loss_rate / niter))


if __name__ == "__main__":
    print("Train a model and convert it into onnx")
    singa_to_onnx(3, True)
    print("Load the onnx model and continue training")
    onnx_to_singa(3, True)
