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


if __name__ == '__main__':

    # prepare training data in numpy array

    # generate the boundary
    f = lambda x: (5 * x + 1)
    bd_x = np.linspace(-1., 1, 200)
    bd_y = f(bd_x)
    # generate the training data
    x = np.random.uniform(-1, 1, 400)
    y = f(x) + 2 * np.random.randn(len(x))
    # convert training data to 2d space
    label = np.asarray([5 * a + 1 > b for (a, b) in zip(x, y)])
    data = np.array([[a, b] for (a, b) in zip(x, y)], dtype=np.float32)

    def to_categorical(y, num_classes):
        '''
        Converts a class vector (integers) to binary class matrix.

        Args
            y: class vector to be converted into a matrix
                (integers from 0 to num_classes).
            num_classes: total number of classes.

        Return
            A binary matrix representation of the input.
        '''
        y = np.array(y, dtype='int')
        n = y.shape[0]
        categorical = np.zeros((n, num_classes))
        categorical[np.arange(n), y] = 1
        return categorical

    label = to_categorical(label, 2).astype(np.float32)
    print('train_data_shape:', data.shape)
    print('train_label_shape:', label.shape)

    inputs = Tensor(data=data)
    target = Tensor(data=label)

    w0 = Tensor(shape=(2, 3), requires_grad=True, stores_grad=True)
    w0.gaussian(0.0, 0.1)
    b0 = Tensor(shape=(1, 3), requires_grad=True, stores_grad=True)
    b0.set_value(0.0)

    w1 = Tensor(shape=(3, 2), requires_grad=True, stores_grad=True)
    w1.gaussian(0.0, 0.1)
    b1 = Tensor(shape=(1, 2), requires_grad=True, stores_grad=True)
    b1.set_value(0.0)

    sgd = optimizer.SGD(0.05)
    # training process
    for i in range(1001):
        x = autograd.matmul(inputs, w0)
        x = autograd.add_bias(x, b0)
        x = autograd.relu(x)
        x = autograd.matmul(x, w1)
        x = autograd.add_bias(x, b1)
        x = autograd.soft_max(x)
        loss = autograd.cross_entropy(x, target)
        in_grads = autograd.backward(loss)

        for param in in_grads:
            sgd.apply(0, in_grads[param], param, '')

        if (i % 100 == 0):
            print('training loss = ', tensor.to_numpy(loss)[0])
