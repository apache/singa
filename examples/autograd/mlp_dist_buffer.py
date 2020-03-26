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
from singa import opt
from singa import device
import numpy as np

if __name__ == "__main__":
    sgd = opt.SGD(0.05)
    sgd = opt.DistOpt(sgd)

    dev = device.create_cuda_gpu_on(sgd.rank_in_local)

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
    if sgd.rank_in_global == 0:
        print("train_data_shape:", data.shape, flush=True)
        print("train_label_shape:", label.shape, flush=True)

    inputs = Tensor(data=data, device=dev)
    target = Tensor(data=label, device=dev)

    w0 = Tensor(shape=(2, 3), device=dev, requires_grad=True, stores_grad=True)
    w0.gaussian(0.0, 0.1)
    b0 = Tensor(shape=(3,), device=dev, requires_grad=True, stores_grad=True)
    b0.set_value(0.0)

    w1 = Tensor(shape=(3, 2), device=dev, requires_grad=True, stores_grad=True)
    w1.gaussian(0.0, 0.1)
    b1 = Tensor(shape=(2,), device=dev, requires_grad=True, stores_grad=True)
    b1.set_value(0.0)

    if sgd.rank_in_global == 0:
        print("finished init inputs", flush=True)
        print("w0:\n", tensor.to_numpy(w0), flush=True)
        print("b0:\n", tensor.to_numpy(b0), flush=True)
        print("w1:\n", tensor.to_numpy(w1), flush=True)
        print("b1:\n", tensor.to_numpy(b1), flush=True)

    # training process
    if sgd.rank_in_global == 0:
        print("start training", flush=True)

    # Buffer the operations
    dev.EnableGraph(True)
    x = autograd.matmul(inputs, w0)
    x = autograd.add_bias(x, b0)
    x = autograd.relu(x)
    x = autograd.matmul(x, w1)
    x = autograd.add_bias(x, b1)
    # x = autograd.softmax(x)
    loss = autograd.softmax_cross_entropy(x, target)
    if sgd.rank_in_global == 0:
        print("start backward", flush=True)
    sgd.backward_and_update(loss)
    dev.EnableGraph(False)

    # exec the buffered ops
    if sgd.rank_in_global == 0:
        print("start executing buffered functions", flush=True)

    import time
    start = time.time()
    for i in range(100001):
        dev.RunGraph()
        if i % 10000 == 0 and sgd.rank_in_global == 0:
            print("training loss = ", tensor.to_numpy(loss)[0], flush=True)

    dev.Sync()
    end = time.time()
    batch_size = 400
    niters = 100000
    throughput = float(sgd.world_size * niters * batch_size) / (end - start)
    titer = (end - start) / float(niters)

    if sgd.rank_in_global == 0:
        print("\nThroughput = {} per second".format(throughput), flush=True)
        print("Total={} {}".format(titer, end - start), flush=True)
