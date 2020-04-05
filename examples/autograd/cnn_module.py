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

from mnist_cnn import accuracy
from mnist_cnn import augmentation
from mnist_cnn import load_dataset
from mnist_cnn import to_categorical
from mnist_cnn import reduce_variable

import numpy as np
import time


class CNN(module.Module):

    def __init__(self, optimizer):
        super(CNN, self).__init__()

        self.conv1 = autograd.Conv2d(1, 20, 5, padding=0)
        self.conv2 = autograd.Conv2d(20, 50, 5, padding=0)
        self.linear1 = autograd.Linear(4 * 4 * 50, 500)
        self.linear2 = autograd.Linear(500, 10)
        self.pooling1 = autograd.MaxPool2d(2, 2, padding=0)
        self.pooling2 = autograd.MaxPool2d(2, 2, padding=0)

        self.optimizer = optimizer

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

    def optim(self, loss, DIST, threshold):
        if DIST:
            self.optimizer.backward_and_update(loss, threshold)
        else:
            self.optimizer.backward_and_update(loss)


def data_partition(dataset_x, dataset_y, rank_in_global, world_size):
    data_per_rank = dataset_x.shape[0] // world_size
    idx_start = rank_in_global * data_per_rank
    idx_end = (rank_in_global + 1) * data_per_rank
    return dataset_x[idx_start:idx_end], dataset_y[idx_start:idx_end]


def train_mnist_cnn(DIST=False, graph=True, sequential=False):

    # Define the hypermeters good for the mnist_cnn
    max_epoch = 10
    batch_size = 64
    sgd = opt.SGD(lr=0.005, momentum=0.9, weight_decay=1e-5)

    device_id = 0
    world_size = 1
    rank_in_global = 0
    IMG_SIZE = 28
    num_classes = 10

    train_x, train_y, test_x, test_y = load_dataset()
    train_y = to_categorical(train_y, num_classes)
    test_y = to_categorical(test_y, num_classes)
    train_x = train_x / 255
    test_x = test_x / 255

    if DIST:
        sgd = opt.DistOpt(sgd)
        world_size = sgd.world_size
        device_id = sgd.rank_in_local
        rank_in_global = sgd.rank_in_global
        train_x, train_y = data_partition(train_x, train_y, rank_in_global,
                                          world_size)
        test_x, test_y = data_partition(test_x, test_y, rank_in_global,
                                        world_size)

    dev = device.create_cuda_gpu_on(device_id)
    dev.SetRandSeed(0)
    np.random.seed(0)

    # construct model
    model = CNN(sgd)
    model.on_device(dev)
    model.graph(graph, sequential)

    tx = tensor.Tensor((batch_size, 1, IMG_SIZE, IMG_SIZE), dev, tensor.float32)
    ty = tensor.Tensor((batch_size, num_classes), dev, tensor.int32)
    num_train_batch = train_x.shape[0] // batch_size
    num_test_batch = test_x.shape[0] // batch_size
    idx = np.arange(train_x.shape[0], dtype=np.int32)

    model = CNN(sgd)
    model.train()
    model.on_device(dev)
    model.graph(graph, sequential)

    # Training and Evaluation Loop
    for epoch in range(max_epoch):
        start_time = time.time()
        np.random.shuffle(idx)

        if rank_in_global == 0:
            print('Starting Epoch %d:' % (epoch))

        # Training Phase
        train_correct = np.zeros(shape=[1], dtype=np.float32)
        test_correct = np.zeros(shape=[1], dtype=np.float32)
        train_loss = np.zeros(shape=[1], dtype=np.float32)

        model.train()
        for b in range(num_train_batch):
            # Generate the patch data in this iteration
            x = train_x[idx[b * batch_size:(b + 1) * batch_size]]
            x = augmentation(x, batch_size)
            y = train_y[idx[b * batch_size:(b + 1) * batch_size]]

            # Copy the patch data into input tensors
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)

            # Train the model
            out = model(tx)
            loss = model.loss(out, ty)
            model.optim(loss, DIST, 50000)

            train_correct += accuracy(tensor.to_numpy(out), y)
            train_loss += tensor.to_numpy(loss)[0]

        if DIST:
            # Reduce the Evaluation Accuracy and Loss from Multiple Devices
            reducer = tensor.Tensor((1,), dev, tensor.float32)
            train_correct = reduce_variable(train_correct, sgd, reducer)
            train_loss = reduce_variable(train_loss, sgd, reducer)

        if rank_in_global == 0:
            print('Training loss = %f, training accuracy = %f' %
                  (train_loss, train_correct /
                   (num_train_batch * batch_size * world_size)),
                  flush=True)

        # Evaluation Phase
        model.eval()
        for b in range(num_test_batch):
            x = test_x[b * batch_size:(b + 1) * batch_size]
            y = test_y[b * batch_size:(b + 1) * batch_size]
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            out_test = model(tx)
            test_correct += accuracy(tensor.to_numpy(out_test), y)

        if DIST:
            # Reduce the Evaulation Accuracy from Multiple Devices
            test_correct = reduce_variable(test_correct, sgd, reducer)

        # Output the Evaluation Accuracy
        if rank_in_global == 0:
            print('Evaluation accuracy = %f, Elapsed Time = %fs' %
                  (test_correct / (num_test_batch * batch_size * world_size),
                   time.time() - start_time),
                  flush=True)


if __name__ == '__main__':

    DIST = False
    graph = True
    sequential = False

    # For distributed training, sequential has better throughput in the current version
    if DIST:
        sequential = True

    train_mnist_cnn(DIST=DIST, graph=graph, sequential=sequential)
