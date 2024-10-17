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

from singa import singa_wrap as singa
from singa import device
from singa import tensor
from singa import opt
import numpy as np
import time
import argparse
from PIL import Image

np_dtype = {"float16": np.float16, "float32": np.float32}

singa_dtype = {"float16": tensor.float16, "float32": tensor.float32}


# Data augmentation
def augmentation(x, batch_size):
    xpad = np.pad(x, [[0, 0], [0, 0], [4, 4], [4, 4]], 'symmetric')
    for data_num in range(0, batch_size):
        offset = np.random.randint(8, size=2)
        x[data_num, :, :, :] = xpad[data_num, :,
                                    offset[0]:offset[0] + x.shape[2],
                                    offset[1]:offset[1] + x.shape[2]]
        if_flip = np.random.randint(2)
        if (if_flip):
            x[data_num, :, :, :] = x[data_num, :, :, ::-1]
    return x


# Calculate accuracy
def accuracy(pred, target):
    # y is network output to be compared with ground truth (int)
    y = np.argmax(pred, axis=1)
    a = y == target
    correct = np.array(a, "int").sum()
    return correct


# Data partition according to the rank
def partition(global_rank, world_size, train_x, train_y, val_x, val_y):
    # Partition training data
    data_per_rank = train_x.shape[0] // world_size
    idx_start = global_rank * data_per_rank
    idx_end = (global_rank + 1) * data_per_rank
    train_x = train_x[idx_start:idx_end]
    train_y = train_y[idx_start:idx_end]

    # Partition evaluation data
    data_per_rank = val_x.shape[0] // world_size
    idx_start = global_rank * data_per_rank
    idx_end = (global_rank + 1) * data_per_rank
    val_x = val_x[idx_start:idx_end]
    val_y = val_y[idx_start:idx_end]
    return train_x, train_y, val_x, val_y


# Function to all reduce NUMPY accuracy and loss from multiple devices
def reduce_variable(variable, dist_opt, reducer):
    reducer.copy_from_numpy(variable)
    dist_opt.all_reduce(reducer.data)
    dist_opt.wait()
    output = tensor.to_numpy(reducer)
    return output


def resize_dataset(x, image_size):
    num_data = x.shape[0]
    dim = x.shape[1]
    X = np.zeros(shape=(num_data, dim, image_size, image_size),
                 dtype=np.float32)
    for n in range(0, num_data):
        for d in range(0, dim):
            X[n, d, :, :] = np.array(Image.fromarray(x[n, d, :, :]).resize(
                (image_size, image_size), Image.BILINEAR),
                                     dtype=np.float32)
    return X


def run(global_rank,
        world_size,
        local_rank,
        max_epoch,
        batch_size,
        model,
        data,
        sgd,
        graph,
        verbosity,
        dist_option='plain',
        spars=None,
        precision='float32'):
    dev = device.create_cuda_gpu_on(
        local_rank)  # need to change to CPU device for CPU-only machines
    dev.SetRandSeed(0)
    np.random.seed(0)

    if data == 'cifar10':
        from data import cifar10
        train_x, train_y, val_x, val_y = cifar10.load()
    elif data == 'cifar100':
        from data import cifar100
        train_x, train_y, val_x, val_y = cifar100.load()
    elif data == 'mnist':
        from data import mnist
        train_x, train_y, val_x, val_y = mnist.load()

    num_channels = train_x.shape[1]
    image_size = train_x.shape[2]
    data_size = np.prod(train_x.shape[1:train_x.ndim]).item()
    num_classes = (np.max(train_y) + 1).item()

    if model == 'resnet':
        from model import resnet
        model = resnet.resnet50(num_channels=num_channels,
                                num_classes=num_classes)
    elif model == 'xceptionnet':
        from model import xceptionnet
        model = xceptionnet.create_model(num_channels=num_channels,
                                         num_classes=num_classes)
    elif model == 'cnn':
        from model import cnn
        model = cnn.create_model(num_channels=num_channels,
                                 num_classes=num_classes)
    elif model == 'alexnet':
        from model import alexnet
        model = alexnet.create_model(num_channels=num_channels,
                                     num_classes=num_classes)
    elif model == 'mlp':
        import os, sys, inspect
        current = os.path.dirname(
            os.path.abspath(inspect.getfile(inspect.currentframe())))
        parent = os.path.dirname(current)
        sys.path.insert(0, parent)
        from mlp import model
        model = model.create_model(data_size=data_size, num_classes=num_classes)

    # For distributed training, sequential has better performance
    if hasattr(sgd, "communicator"):
        DIST = True
        sequential = True
    else:
        DIST = False
        sequential = False

    if DIST:
        train_x, train_y, val_x, val_y = partition(global_rank, world_size,
                                                   train_x, train_y, val_x,
                                                   val_y)

    if model.dimension == 4:
        tx = tensor.Tensor(
            (batch_size, num_channels, model.input_size, model.input_size), dev,
            singa_dtype[precision])
    elif model.dimension == 2:
        tx = tensor.Tensor((batch_size, data_size), dev, singa_dtype[precision])
        np.reshape(train_x, (train_x.shape[0], -1))
        np.reshape(val_x, (val_x.shape[0], -1))

    ty = tensor.Tensor((batch_size,), dev, tensor.int32)
    num_train_batch = train_x.shape[0] // batch_size
    num_val_batch = val_x.shape[0] // batch_size
    idx = np.arange(train_x.shape[0], dtype=np.int32)

    # Attach model to graph
    model.set_optimizer(sgd)
    model.compile([tx], is_train=True, use_graph=graph, sequential=sequential)
    dev.SetVerbosity(verbosity)

    # Training and evaluation loop
    for epoch in range(max_epoch):
        start_time = time.time()
        np.random.shuffle(idx)

        if global_rank == 0:
            print('Starting Epoch %d:' % (epoch))

        # Training phase
        train_correct = np.zeros(shape=[1], dtype=np.float32)
        test_correct = np.zeros(shape=[1], dtype=np.float32)
        train_loss = np.zeros(shape=[1], dtype=np.float32)

        model.train()
        for b in range(num_train_batch):
            # if b % 100 == 0:
            #     print ("b: \n", b)
            # Generate the patch data in this iteration
            x = train_x[idx[b * batch_size:(b + 1) * batch_size]]
            if model.dimension == 4:
                x = augmentation(x, batch_size)
                if (image_size != model.input_size):
                    x = resize_dataset(x, model.input_size)
            x = x.astype(np_dtype[precision])
            y = train_y[idx[b * batch_size:(b + 1) * batch_size]]

            # Copy the patch data into input tensors
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)

            # Train the model
            out, loss = model(tx, ty, dist_option, spars)
            train_correct += accuracy(tensor.to_numpy(out), y)
            train_loss += tensor.to_numpy(loss)[0]

        if DIST:
            # Reduce the evaluation accuracy and loss from multiple devices
            reducer = tensor.Tensor((1,), dev, tensor.float32)
            train_correct = reduce_variable(train_correct, sgd, reducer)
            train_loss = reduce_variable(train_loss, sgd, reducer)

        if global_rank == 0:
            print('Training loss = %f, training accuracy = %f' %
                  (train_loss, train_correct /
                   (num_train_batch * batch_size * world_size)),
                  flush=True)

        # Evaluation phase
        model.eval()
        for b in range(num_val_batch):
            x = val_x[b * batch_size:(b + 1) * batch_size]
            if model.dimension == 4:
                if (image_size != model.input_size):
                    x = resize_dataset(x, model.input_size)
            x = x.astype(np_dtype[precision])
            y = val_y[b * batch_size:(b + 1) * batch_size]
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            out_test = model(tx)
            test_correct += accuracy(tensor.to_numpy(out_test), y)

        if DIST:
            # Reduce the evaulation accuracy from multiple devices
            test_correct = reduce_variable(test_correct, sgd, reducer)

        # Output the evaluation accuracy
        if global_rank == 0:
            print('Evaluation accuracy = %f, Elapsed Time = %fs' %
                  (test_correct / (num_val_batch * batch_size * world_size),
                   time.time() - start_time),
                  flush=True)

    dev.PrintTimeProfiling()


if __name__ == '__main__':
    # Use argparse to get command config: max_epoch, model, data, etc., for single gpu training
    parser = argparse.ArgumentParser(
        description='Training using the autograd and graph.')
    parser.add_argument(
        'model',
        choices=['cnn', 'resnet', 'xceptionnet', 'mlp', 'alexnet'],
        default='cnn')
    parser.add_argument('data',
                        choices=['mnist', 'cifar10', 'cifar100'],
                        default='mnist')
    parser.add_argument('-p',
                        choices=['float32', 'float16'],
                        default='float32',
                        dest='precision')
    parser.add_argument('-m',
                        '--max-epoch',
                        default=100,
                        type=int,
                        help='maximum epochs',
                        dest='max_epoch')
    parser.add_argument('-b',
                        '--batch-size',
                        default=64,
                        type=int,
                        help='batch size',
                        dest='batch_size')
    parser.add_argument('-l',
                        '--learning-rate',
                        default=0.005,
                        type=float,
                        help='initial learning rate',
                        dest='lr')
    # Determine which gpu to use
    parser.add_argument('-i',
                        '--device-id',
                        default=0,
                        type=int,
                        help='which GPU to use',
                        dest='device_id')
    parser.add_argument('-g',
                        '--disable-graph',
                        default='True',
                        action='store_false',
                        help='disable graph',
                        dest='graph')
    parser.add_argument('-v',
                        '--log-verbosity',
                        default=0,
                        type=int,
                        help='logging verbosity',
                        dest='verbosity')

    args = parser.parse_args()

    sgd = opt.SGD(lr=args.lr,
                  momentum=0.9,
                  weight_decay=1e-5,
                  dtype=singa_dtype[args.precision])
    run(0,
        1,
        args.device_id,
        args.max_epoch,
        args.batch_size,
        args.model,
        args.data,
        sgd,
        args.graph,
        args.verbosity,
        precision=args.precision)
