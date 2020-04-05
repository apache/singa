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
from singa import opt
from singa import device
from singa import tensor
import numpy as np
import time
import argparse
from PIL import Image


# Data Augmentation
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


# Calculate Accuracy
def accuracy(pred, target):
    # y is binary array for ground truth
    y = np.argmax(pred, axis=1)
    # t is integer for ground truth, the dim of t is (batch, 1)
    t = np.max(target, axis=1)
    a = y == t
    correct = np.array(a, "int").sum()
    # print(correct)
    return correct


# Data partition according to the rank
def partition(worker_id, num_workers, train_x, train_y, val_x, val_y):
    # Partition training data
    data_per_rank = train_x.shape[0] // num_workers
    idx_start = worker_id * data_per_rank
    idx_end = (worker_id + 1) * data_per_rank
    train_x = train_x[idx_start:idx_end]
    train_y = train_y[idx_start:idx_end]
    # Partition evaluation data
    data_per_rank = val_x.shape[0] // num_workers
    idx_start = worker_id * data_per_rank
    idx_end = (worker_id + 1) * data_per_rank
    val_x = val_x[idx_start:idx_end]
    val_y = val_y[idx_start:idx_end]
    return train_x, train_y, val_x, val_y


# Function to all reduce NUMPY Accuracy and Loss from Multiple Devices
def reduce_variable(variable, dist_opt, reducer):
    reducer.copy_from_numpy(variable)
    dist_opt.all_reduce(reducer.data)
    dist_opt.wait()
    output = tensor.to_numpy(reducer)
    return output


def resize_dataset(x, image_size):
    num_data = x.shape[0]
    dim = x.shape[1]
    X = np.zeros(shape=(num_data, dim, image_size, image_size), dtype=np.float32)
    for n in range(0, num_data):
        for d in range(0, dim):
            X[n, d, :, :] = np.array(Image.fromarray(x[n, d, :, :]).resize(
                (image_size, image_size), Image.BILINEAR),
                                     dtype=np.float32)
    return X


def run(worker_id, num_workers, device_id, max_epoch, batch_size, model, data,
        sgd):
    dev = device.create_cuda_gpu_on(device_id)
    dev.SetRandSeed(0)
    np.random.seed(0)

    if model == 'resnet':
        from model import resnet
        model = resnet.resnet50()
    elif model == 'xceptionnet':
        from model import xceptionnet
        model = xceptionnet.create_model()
    elif model == 'cnn':
        from model import cnn
        model = cnn.create_model()

    if data == 'cifar10':
        from data import cifar10
        train_x, train_y, val_x, val_y = cifar10.load()
    elif data == 'mnist':
        from data import mnist
        train_x, train_y, val_x, val_y = mnist.load()

    if hasattr(sgd, "communicator"):
        DIST = True
    else:
        DIST = False

    if DIST:
        train_x, train_y, val_x, val_y = partition(worker_id, num_workers,
                                                   train_x, train_y, val_x,
                                                   val_y)
    '''
    # check dataset shape correctness
    if worker_id == 0:
        print("Check the shape of dataset:")
        print(train_x.shape)
        print(train_y.shape)
    '''

    num_channels = train_x.shape[1]
    image_size = train_x.shape[2]

    tx = tensor.Tensor(
        (batch_size, num_channels, model.input_size, model.input_size), dev,
        tensor.float32)
    ty = tensor.Tensor((batch_size,), dev, tensor.int32)
    num_train_batch = train_x.shape[0] // batch_size
    num_val_batch = val_x.shape[0] // batch_size
    idx = np.arange(train_x.shape[0], dtype=np.int32)

    # attached model to graph
    graph = True

    # For distributed training, sequential gives better performance
    if DIST:
        sequential = True
    else:
        sequential = False

    model.on_device(dev)
    model.set_optimizer(sgd)
    model.graph(graph, sequential)

    # Training and Evaluation Loop
    for epoch in range(max_epoch):
        start_time = time.time()
        np.random.shuffle(idx)

        if worker_id == 0:
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
            if (image_size != model.input_size):
                x = resize_dataset(x, model.input_size)
            y = train_y[idx[b * batch_size:(b + 1) * batch_size]]

            # Copy the patch data into input tensors
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)

            # Train the model
            out = model(tx)
            loss = model.loss(out, ty)
            model.optim(loss)
            train_correct += accuracy(tensor.to_numpy(out),y)
            train_loss += tensor.to_numpy(loss)[0]

        if DIST:
            # Reduce the Evaluation Accuracy and Loss from Multiple Devices
            reducer = tensor.Tensor((1,), dev, tensor.float32)
            train_correct = reduce_variable(train_correct, sgd, reducer)
            train_loss = reduce_variable(train_loss, sgd, reducer)

        if worker_id == 0:
            print('Training loss = %f, training accuracy = %f' %
                  (train_loss, train_correct /
                   (num_train_batch * batch_size * num_workers)),
                  flush=True)

        # Evaluation Phase
        model.eval()
        for b in range(num_val_batch):
            x = val_x[b * batch_size:(b + 1) * batch_size]
            if (image_size != model.input_size):
                x = resize_dataset(x, model.input_size)
            y = val_y[b * batch_size:(b + 1) * batch_size]
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            out_test = model(tx)
            test_correct += accuracy(tensor.to_numpy(out_test),y)

        if DIST:
            # Reduce the Evaulation Accuracy from Multiple Devices
            test_correct = reduce_variable(test_correct, sgd, reducer)

        # Output the Evaluation Accuracy
        if worker_id == 0:
            print('Evaluation accuracy = %f, Elapsed Time = %fs' %
                  (test_correct / (num_val_batch * batch_size * num_workers),
                   time.time() - start_time),
                  flush=True)


if __name__ == '__main__':
    # use argparse to get command config: max_epoch, model, data, etc. for single gpu training
    parser = argparse.ArgumentParser(
        description='Training using the autograd and graph.')
    parser.add_argument('model',
                        choices=['resnet', 'xceptionnet', 'cnn'],
                        default='cnn')
    parser.add_argument('data', choices=['cifar10', 'mnist'], default='mnist')
    parser.add_argument('--epoch',
                        '--max-epoch',
                        default=10,
                        type=int,
                        help='maximum epochs',
                        dest='max_epoch')
    parser.add_argument('--bs',
                        '--batch-size',
                        default=64,
                        type=int,
                        help='batch size',
                        dest='batch_size')
    parser.add_argument('--lr',
                        '--learning-rate',
                        default=0.005,
                        type=float,
                        help='initial learning rate',
                        dest='lr')

    args = parser.parse_args()

    sgd = opt.SGD(lr=args.lr, momentum=0.9, weight_decay=1e-5)
    run(0, 1, 0, args.max_epoch, args.batch_size, args.model, args.data, sgd)
