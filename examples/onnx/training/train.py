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

import sys, os
import json
from singa import singa_wrap as singa
from singa import opt
from singa import device
from singa import tensor
from singa import sonnx
from singa import layer
from singa import autograd
import numpy as np
import time
import argparse
from PIL import Image
import onnx
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
sys.path.append(os.path.dirname(__file__) + '/../../cnn')
sys.path.append(os.path.dirname(__file__) + '/..')
from utils import download_model


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
    # y is network output to be compared with ground truth (int)
    y = np.argmax(pred, axis=1)
    a = y == target
    correct = np.array(a, "int").sum()
    # print(correct)
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
    X = np.zeros(shape=(num_data, dim, image_size, image_size),
                 dtype=np.float32)
    for n in range(0, num_data):
        for d in range(0, dim):
            X[n, d, :, :] = np.array(Image.fromarray(x[n, d, :, :]).resize(
                (image_size, image_size), Image.BILINEAR),
                                     dtype=np.float32)
    return X


class MyModel(sonnx.SONNXModel):

    def __init__(self,
                 onnx_model,
                 num_classes=10,
                 num_channels=3,
                 last_layers=-1,
                 in_dim=1000):
        super(MyModel, self).__init__(onnx_model)
        self.num_classes = num_classes
        self.input_size = 224
        self.dimension = 4
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.last_layers = last_layers
        self.linear = layer.Linear(in_dim, num_classes)

    def forward(self, *x):
        y = super(MyModel, self).forward(*x, last_layers=self.last_layers)[0]
        y = self.linear(y)
        return y

    def train_one_batch(self, x, y, dist_option, spars):
        out = self.forward(x)
        loss = autograd.softmax_cross_entropy(out, y)
        if dist_option == 'fp32':
            self.optimizer.backward_and_update(loss)
        elif dist_option == 'fp16':
            self.optimizer.backward_and_update_half(loss)
        elif dist_option == 'partialUpdate':
            self.optimizer.backward_and_partial_update(loss)
        elif dist_option == 'sparseTopK':
            self.optimizer.backward_and_sparse_update(loss,
                                                      topK=True,
                                                      spars=spars)
        elif dist_option == 'sparseThreshold':
            self.optimizer.backward_and_sparse_update(loss,
                                                      topK=False,
                                                      spars=spars)
        return out, loss

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer


def run(global_rank,
        world_size,
        local_rank,
        max_epoch,
        batch_size,
        model_config,
        data,
        sgd,
        graph,
        verbosity,
        dist_option='fp32',
        spars=None):
    dev = device.create_cuda_gpu_on(local_rank)
    dev.SetRandSeed(0)
    np.random.seed(0)

    if data == 'cifar10':
        from data import cifar10
        train_x, train_y, val_x, val_y = cifar10.load()
    elif data == 'cifar100':
        from data import cifar100
        train_x, train_y, val_x, val_y = cifar100.load()

    num_channels = train_x.shape[1]
    image_size = train_x.shape[2]
    data_size = np.prod(train_x.shape[1:train_x.ndim]).item()
    num_classes = (np.max(train_y) + 1).item()

    # read and make onnx model
    download_model(model_config['url'])
    onnx_model = onnx.load(os.path.join('/tmp', model_config['path']))
    model = MyModel(onnx_model,
                    num_channels=num_channels,
                    num_classes=num_classes,
                    last_layers=model_config['last_layers'],
                    in_dim=model_config['last_layers_dim'])

    # For distributed training, sequential gives better performance
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
    '''
    # check dataset shape correctness
    if global_rank == 0:
        print("Check the shape of dataset:")
        print(train_x.shape)
        print(train_y.shape)
    '''

    if model.dimension == 4:
        tx = tensor.Tensor(
            (batch_size, num_channels, model.input_size, model.input_size), dev,
            tensor.float32)
    elif model.dimension == 2:
        tx = tensor.Tensor((batch_size, data_size), dev, tensor.float32)
        np.reshape(train_x, (train_x.shape[0], -1))
        np.reshape(val_x, (val_x.shape[0], -1))

    ty = tensor.Tensor((batch_size,), dev, tensor.int32)
    num_train_batch = train_x.shape[0] // batch_size
    num_val_batch = val_x.shape[0] // batch_size
    idx = np.arange(train_x.shape[0], dtype=np.int32)

    # attached model to graph
    model.set_optimizer(sgd)
    model.compile([tx], is_train=True, use_graph=graph, sequential=sequential)
    dev.SetVerbosity(verbosity)

    # Training and Evaluation Loop
    for epoch in range(max_epoch):
        start_time = time.time()
        np.random.shuffle(idx)

        if global_rank == 0:
            print('Starting Epoch %d:' % (epoch))

        # Training Phase
        train_correct = np.zeros(shape=[1], dtype=np.float32)
        test_correct = np.zeros(shape=[1], dtype=np.float32)
        train_loss = np.zeros(shape=[1], dtype=np.float32)

        model.train()
        for b in tqdm(range(num_train_batch)):
            # Generate the patch data in this iteration
            x = train_x[idx[b * batch_size:(b + 1) * batch_size]]
            if model.dimension == 4:
                x = augmentation(x, batch_size)
                if (image_size != model.input_size):
                    x = resize_dataset(x, model.input_size)
            y = train_y[idx[b * batch_size:(b + 1) * batch_size]]

            # Copy the patch data into input tensors
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)

            # Train the model
            out, loss = model(tx, ty, dist_option, spars)
            train_correct += accuracy(tensor.to_numpy(out), y)
            train_loss += tensor.to_numpy(loss)[0]

        if DIST:
            # Reduce the Evaluation Accuracy and Loss from Multiple Devices
            reducer = tensor.Tensor((1,), dev, tensor.float32)
            train_correct = reduce_variable(train_correct, sgd, reducer)
            train_loss = reduce_variable(train_loss, sgd, reducer)

        if global_rank == 0:
            print('Training loss = %f, training accuracy = %f' %
                  (train_loss, train_correct /
                   (num_train_batch * batch_size * world_size)),
                  flush=True)

        # Evaluation Phase
        model.eval()
        for b in tqdm(range(num_val_batch)):
            x = val_x[b * batch_size:(b + 1) * batch_size]
            if model.dimension == 4:
                if (image_size != model.input_size):
                    x = resize_dataset(x, model.input_size)
            y = val_y[b * batch_size:(b + 1) * batch_size]
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            out_test = model(tx)
            test_correct += accuracy(tensor.to_numpy(out_test), y)

        if DIST:
            # Reduce the Evaulation Accuracy from Multiple Devices
            test_correct = reduce_variable(test_correct, sgd, reducer)

        # Output the Evaluation Accuracy
        if global_rank == 0:
            print('Evaluation accuracy = %f, Elapsed Time = %fs' %
                  (test_correct / (num_val_batch * batch_size * world_size),
                   time.time() - start_time),
                  flush=True)

    dev.PrintTimeProfiling()


def loss(out, y):
    return autograd.softmax_cross_entropy(out, y)


if __name__ == '__main__':

    with open(os.path.join(os.path.dirname(__file__),
                           'model.json')) as json_file:
        model_config = json.load(json_file)

    # use argparse to get command config: max_epoch, model, data, etc. for single gpu training
    parser = argparse.ArgumentParser(
        description='Training using the autograd and graph.')
    parser.add_argument('--model',
                        choices=list(model_config.keys()),
                        help='please refer to the models.json for more details',
                        default='resnet18v1')
    parser.add_argument('--data',
                        choices=['cifar10', 'cifar100'],
                        default='cifar10')
    parser.add_argument('--epoch',
                        '--max-epoch',
                        default=10,
                        type=int,
                        help='maximum epochs',
                        dest='max_epoch')
    parser.add_argument('--bs',
                        '--batch-size',
                        default=32,
                        type=int,
                        help='batch size',
                        dest='batch_size')
    parser.add_argument('--lr',
                        '--learning-rate',
                        default=0.005,
                        type=float,
                        help='initial learning rate',
                        dest='lr')
    # determine which gpu to use
    parser.add_argument('--id',
                        '--device-id',
                        default=0,
                        type=int,
                        help='which GPU to use',
                        dest='device_id')
    parser.add_argument('--no-graph',
                        '--disable-graph',
                        default='True',
                        action='store_false',
                        help='disable graph',
                        dest='graph')
    parser.add_argument('--verbosity',
                        '--log-verbosity',
                        default=1,
                        type=int,
                        help='logging verbosity',
                        dest='verbosity')

    args = parser.parse_args()

    sgd = opt.SGD(lr=args.lr, momentum=0.9, weight_decay=1e-5)
    run(0, 1, args.device_id, args.max_epoch, args.batch_size,
        model_config[args.model], args.data, sgd, args.graph, args.verbosity)
