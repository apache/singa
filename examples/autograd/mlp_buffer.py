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

from singa import tensor
from singa.tensor import Tensor
from singa import autograd
from singa import optimizer
from singa import device
import numpy as np


if __name__ == "__main__":
    dev = device.get_default_device()

    autograd.training = True
    np.random.seed(0)
    
    # prepare training data in numpy array

    # generate the boundary
    f = lambda x: (5 * x + 1)
    bd_x = np.linspace(-1.0, 1, 200)
    bd_y = f(bd_x)
    # generate the training data
    x = np.random.uniform(-1, 1, 400)
    y = f(x) + 2 * np.random.randn(len(x))
    # convert training data to 2d space
    label = np.asarray([5 * a + 1 > b for (a, b) in zip(x, y)])
    data = np.array([[a, b] for (a, b) in zip(x, y)], dtype=np.float32)

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

    inputs = Tensor(data=data, device=dev)
    target = Tensor(data=label, device=dev)

    w0 = Tensor(shape=(2, 3), device=dev, requires_grad=True, stores_grad=True)
    w0.gaussian(0.0, 0.1)
    b0 = Tensor(shape=(1, 3), device=dev, requires_grad=True, stores_grad=True)
    b0.set_value(0.0)

    w1 = Tensor(shape=(3, 2), device=dev, requires_grad=True, stores_grad=True)
    w1.gaussian(0.0, 0.1)
    b1 = Tensor(shape=(1, 2), device=dev, requires_grad=True, stores_grad=True)
    b1.set_value(0.0)

    print("finished init inputs")
    print("w0:\n", tensor.to_numpy(w0))
    print("b0:\n", tensor.to_numpy(b0))
    print("w1:\n", tensor.to_numpy(w1))
    print("b1:\n", tensor.to_numpy(b1))

    sgd = optimizer.SGD(0.05)

    # training process
    print("start training")

    # Buffer the operations
    dev.SetBufferFlag(True)
    x = autograd.matmul(inputs, w0)
    x = autograd.add_bias(x, b0)
    x = autograd.relu(x)
    x = autograd.matmul(x, w1)
    x = autograd.add_bias(x, b1)
    # x = autograd.softmax(x)
    loss = autograd.softmax_cross_entropy(x, target)
    print("start backward")
    for p, gp in autograd.backward(loss):
        sgd.apply(0, gp, p, "")
    dev.SetBufferFlag(False)

    # exec the buffered ops
    print("start executing buffered functions")
    for i in range(1001):
        dev.ExecBuffOps()
        if i % 100 == 0:
            print("training loss = ", tensor.to_numpy(loss)[0])
