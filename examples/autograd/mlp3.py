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
import numpy as np
from singa import device

if __name__ == "__main__":
    # dev = device.get_default_device()
    dev = device.create_cuda_gpu_on(0)

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

    dev.SetBufferFlag(False)
    inputs = Tensor(data=data, device=dev)
    target = Tensor(data=label, device=dev)

    w0 = Tensor(shape=(2, 3), device=dev, requires_grad=True, stores_grad=True)
    w0.set_value(10.0)
    b0 = Tensor(shape=(1, 3), device=dev, requires_grad=True, stores_grad=True)
    b0.set_value(1.0)

    w1 = Tensor(shape=(3, 2), device=dev, requires_grad=True, stores_grad=True)
    w1.set_value(10.0)
    b1 = Tensor(shape=(1, 2), device=dev, requires_grad=True, stores_grad=True)
    b1.set_value(0.0)

    print("finished init inputs")

    # print("data:", data[0])
    # print("inputs", tensor.to_numpy(inputs))
    # print("label", label[0])
    # print("target", tensor.to_numpy(target))
    print("w0:\n", tensor.to_numpy(w0))
    print("b0:\n", tensor.to_numpy(b0))
    print("w1:\n", tensor.to_numpy(w1))
    print("b1:\n", tensor.to_numpy(b1))

    sgd = optimizer.SGD(0.05)

    dev.SetBufferFlag(True)
    print("start farward propagation")

    x = autograd.matmul(inputs, w0)
    x = autograd.add_bias(x, b0)
    x = autograd.relu(x)
    x = autograd.matmul(x, w1)
    x = autograd.add_bias(x, b1)
    loss = autograd.softmax_cross_entropy(x, target)

    dev.ExecBuffOps()
    
    print("start backward propagation")
    
    tg = []
    for p, gp in autograd.backward(loss):
        tg.append(gp)
        sgd.apply(0, gp, p, "")

    dev.ExecBuffOps()

    print("finished training")
    dev.SetBufferFlag(False)

    print("loss:\n", tensor.to_numpy(loss))
    print("w0:\n", tensor.to_numpy(w0))
    print("b0:\n", tensor.to_numpy(b0))
    print("w1:\n", tensor.to_numpy(w1))
    print("b1:\n", tensor.to_numpy(b1))
    for i, t in enumerate(tg):
        print("gradient %d:%s\n"%(i, t.name), tensor.to_numpy(t))
