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

from singa.tensor import Tensor
from singa import tensor
from singa import device
from singa import autograd
from singa import opt
from singa import sonnx
import numpy as np

# prepare training data in numpy array
# generate the boundary
f = lambda x: (5 * x + 1)
bd_x = np.linspace(-1.0, 1, 200)
bd_y = f(bd_x)
# generate the training data
x = np.random.uniform(-1, 1, 300)
y = f(x) + 2 * np.random.randn(len(x))
# convert training data to 2d space
label = np.asarray([5 * a + 1 > b for (a, b) in zip(x, y)])
data = np.array([[a, b] for (a, b) in zip(x, y)], dtype=np.float32)
autograd.training = True


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
    return categorical


label = to_categorical(label, 2).astype(np.float32)
print("train_data_shape:", data.shape)
print("train_label_shape:", label.shape)


def accuracy(pred, target):
    y = np.argmax(pred, axis=1)
    t = np.argmax(target, axis=1)
    a = y == t
    return np.array(a, "int").sum() / float(len(t))


def singa_to_onnx(niter, use_cpu=False):
    if use_cpu:
        print("Using CPU")
        dev = device.get_default_device()
    else:
        print("Using GPU")
        dev = device.create_cuda_gpu()
    inputs = Tensor(
        data=data,
        device=dev,
        requires_grad=False,
        stores_grad=False,
        name="input",
    )
    target = Tensor(
        data=label,
        device=dev,
        requires_grad=False,
        stores_grad=False,
        name="target",
    )

    w0 = Tensor(shape=(2, 3), device=dev, requires_grad=True, stores_grad=True)
    w0.gaussian(0.0, 0.1)
    b0 = Tensor(shape=(3,), device=dev, requires_grad=True, stores_grad=True)
    b0.set_value(0.0)

    w1 = Tensor(shape=(3, 2), device=dev, requires_grad=True, stores_grad=True)
    w1.gaussian(0.0, 0.1)
    b1 = Tensor(shape=(2,), device=dev, requires_grad=True, stores_grad=True)
    b1.set_value(0.0)

    sgd = opt.SGD(0.1)
    # training process
    for i in range(100):
        x = autograd.matmul(inputs, w0)
        x = autograd.add_bias(x, b0)
        x = autograd.relu(x)
        x = autograd.matmul(x, w1)
        x = autograd.add_bias(x, b1)
        loss = autograd.softmax_cross_entropy(x, target)
        for p, gp in autograd.backward(loss):
            sgd.update(p, gp)

        print("training loss = ", tensor.to_numpy(loss)[0])
    sonnx.export([inputs], [x], file_path="mlp.onnx")


def onnx_to_singa(niter, use_cpu=False):
    if use_cpu:
        print("Using CPU")
        dev = device.get_default_device()
    else:
        print("Using GPU")
        dev = device.create_cuda_gpu()
    model = sonnx.load("mlp.onnx")
    backend = sonnx.prepare(model, device=dev)
    sgd = opt.SGD(0.1)
    inputs = Tensor(
        data=data,
        device=dev,
        requires_grad=False,
        stores_grad=False,
        name="input",
    )
    target = Tensor(
        data=label,
        device=dev,
        requires_grad=False,
        stores_grad=False,
        name="target",
    )

    for i in range(100):
        y = backend.run([inputs])[0]
        loss = autograd.softmax_cross_entropy(y, target)
        for p, gp in autograd.backward(loss):
            sgd.update(p, gp)
        loss_rate = tensor.to_numpy(loss)[0]
        accuracy_rate = accuracy(tensor.to_numpy(y), label)

        print("Iter {}, accurate={}, loss={}".format(i, accuracy_rate, loss_rate))


if __name__ == "__main__":
    singa_to_onnx(3, True)
    onnx_to_singa(3, True)
