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
from singa.tensor import Tensor
import numpy as np
import time


class MLP(module.Module):

    def __init__(self, optimizer):
        super(MLP, self).__init__()

        self.w0 = Tensor(shape=(2, 3), requires_grad=True, stores_grad=True)
        self.w0.gaussian(0.0, 0.1)
        self.b0 = Tensor(shape=(3,), requires_grad=True, stores_grad=True)
        self.b0.set_value(0.0)

        self.w1 = Tensor(shape=(3, 2), requires_grad=True, stores_grad=True)
        self.w1.gaussian(0.0, 0.1)
        self.b1 = Tensor(shape=(2,), requires_grad=True, stores_grad=True)
        self.b1.set_value(0.0)

        self.optimizer = optimizer

    def forward(self, inputs):
        x = autograd.matmul(inputs, self.w0)
        x = autograd.add_bias(x, self.b0)
        x = autograd.relu(x)
        x = autograd.matmul(x, self.w1)
        x = autograd.add_bias(x, self.b1)
        return x

    def loss(self, out, target):
        return autograd.softmax_cross_entropy(out, target)

    def optim(self, loss):
        return self.optimizer.backward_and_update(loss)


def to_categorical(y, num_classes):
    """
    Converts a class vector (integers) to binary class matrix.

    Args:
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.

    Return:
        A binary matrix representation of the input.
    """
    y = np.array(y, dtype="int")
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical


def generate_data(num=400):
    # generate the boundary
    f = lambda x: (5 * x + 1)

    # generate the training data
    x = np.random.uniform(-1, 1, num)
    y = f(x) + 2 * np.random.randn(len(x))

    # convert training data to 2d space
    label = np.asarray([5 * a + 1 > b for (a, b) in zip(x, y)])
    data = np.array([[a, b] for (a, b) in zip(x, y)], dtype=np.float32)

    label = to_categorical(label, 2).astype(np.float32)

    return data, label


def train_mlp(DIST=False, graph=True, sequential=False):

    # Define the hypermeters good for the train_mlp
    niters = 10000
    batch_size = 64
    sgd = opt.SGD(lr=0.05)

    local_rank = 0
    world_size = 1
    global_rank = 0

    if DIST:
        sgd = opt.DistOpt(sgd)
        world_size = sgd.world_size
        local_rank = sgd.local_rank
        global_rank = sgd.global_rank

    dev = device.create_cuda_gpu_on(local_rank)

    data, label = generate_data(num=400)
    inputs = Tensor(data=data)
    target = Tensor(data=label)

    model = MLP(sgd)
    model.train()
    model.on_device(dev)
    model.graph(graph, sequential)

    dev.Sync()
    start = time.time()
    for i in range(niters):
        out = model(inputs)
        loss = model.loss(out, target)
        model.optim(loss)

        if i % (niters / 10) == 0 and global_rank == 0:
            print("training loss = ", tensor.to_numpy(loss)[0], flush=True)

    dev.Sync()
    end = time.time()
    titer = (end - start) / float(niters)
    throughput = float(niters * batch_size * world_size) / (end - start)
    if global_rank == 0:
        print("Throughput = {} per second".format(throughput), flush=True)
        print("Total Time={}".format(end - start), flush=True)
        print("Total={}".format(titer), flush=True)


if __name__ == "__main__":

    DIST = False
    graph = True
    sequential = False

    # For distributed training, sequential has better throughput in the current version
    if DIST:
        sequential = True

    train_mlp(DIST=DIST, graph=graph, sequential=sequential)
