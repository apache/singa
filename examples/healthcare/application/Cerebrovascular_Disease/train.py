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

from singa import device
from singa import tensor
from singa import opt
import numpy as np
import time
import argparse
import sys
sys.path.append("../../..")
from healthcare.data import cerebrovascular
from healthcare.models import cerebrovascular_net

np_dtype = {"float16": np.float16, "float32": np.float32}

singa_dtype = {"float16": tensor.float16, "float32": tensor.float32}


def accuracy(pred, target):
    y = np.argmax(pred, axis=1)
    a = y == target
    correct = np.array(a, "int").sum()
    return correct


def partition(global_rank, world_size, train_x, train_y, val_x, val_y):
    data_per_rank = train_x.shape[0] // world_size
    idx_start = global_rank * data_per_rank
    idx_end = (global_rank + 1) * data_per_rank
    train_x = train_x[idx_start:idx_end]
    train_y = train_y[idx_start:idx_end]

    data_per_rank = val_x.shape[0] // world_size
    idx_start = global_rank * data_per_rank
    idx_end = (global_rank + 1) * data_per_rank
    val_x = val_x[idx_start:idx_end]
    val_y = val_y[idx_start:idx_end]
    return train_x, train_y, val_x, val_y


def reduce_variable(variable, dist_opt, reducer):
    reducer.copy_from_numpy(variable)
    dist_opt.all_reduce(reducer.data)
    dist_opt.wait()
    output = tensor.to_numpy(reducer)
    return output


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
        precision='float32',
        dir_path=None):
    dev = device.create_cpu_device()
    dev.SetRandSeed(0)
    np.random.seed(0)

    if data == 'cerebrovascular':
        train_x, train_y, val_x, val_y = cerebrovascular.load(dir_path)
    elif data == 'mnist' or data == 'cifar10' or data == 'cifar100':
        raise ValueError("Only 'cerebrovascular' dataset (2D table data) is supported with MLP model.")

    data_size = train_x.shape[1]
    num_classes = int(np.max(train_y) + 1)

    if model == 'cerebrovascularnet':
        model = cerebrovascular_net.create_model(data_size=data_size,
                                          num_classes=num_classes)
    else:
        print('Wrong model!')
        sys.exit(0)

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

    tx = tensor.Tensor((batch_size, data_size), dev, singa_dtype[precision])
    ty = tensor.Tensor((batch_size,), dev, tensor.int32)

    num_train_batch = train_x.shape[0] // batch_size
    num_val_batch = val_x.shape[0] // batch_size
    idx = np.arange(train_x.shape[0], dtype=np.int32)

    model.set_optimizer(sgd)
    model.compile([tx], is_train=True, use_graph=graph, sequential=sequential)
    dev.SetVerbosity(verbosity)

    for epoch in range(max_epoch):
        start_time = time.time()
        np.random.shuffle(idx)

        if global_rank == 0:
            print('Starting Epoch %d:' % epoch)

        train_correct = np.zeros(shape=[1], dtype=np.float32)
        test_correct = np.zeros(shape=[1], dtype=np.float32)
        train_loss = np.zeros(shape=[1], dtype=np.float32)

        model.train()
        for b in range(num_train_batch):
            x = train_x[idx[b * batch_size:(b + 1) * batch_size]]
            y = train_y[idx[b * batch_size:(b + 1) * batch_size]]

            x = x.astype(np_dtype[precision])
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)

            out, loss = model(tx, ty, dist_option, spars)
            train_correct += accuracy(tensor.to_numpy(out), y)
            train_loss += tensor.to_numpy(loss)[0]

        if DIST:
            reducer = tensor.Tensor((1,), dev, tensor.float32)
            train_correct = reduce_variable(train_correct, sgd, reducer)
            train_loss = reduce_variable(train_loss, sgd, reducer)

        if global_rank == 0:
            print('Training loss = %f, training accuracy = %f' %
                  (train_loss, train_correct /
                   (num_train_batch * batch_size * world_size)),
                  flush=True)

        model.eval()
        for b in range(num_val_batch):
            x = val_x[b * batch_size:(b + 1) * batch_size]
            y = val_y[b * batch_size:(b + 1) * batch_size]

            x = x.astype(np_dtype[precision])
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)

            out_test = model(tx)
            test_correct += accuracy(tensor.to_numpy(out_test), y)

        if DIST:
            test_correct = reduce_variable(test_correct, sgd, reducer)

        if global_rank == 0:
            print('Evaluation accuracy = %f, Elapsed Time = %fs' %
                  (test_correct / (num_val_batch * batch_size * world_size),
                   time.time() - start_time),
                  flush=True)

    dev.PrintTimeProfiling()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training using the autograd and graph.')
    parser.add_argument(
        'model',
        choices=['cnn', 'resnet', 'xceptionnet', 'mlp', 'alexnet', 'cerebrovascularnet'],
        default='cerebrovascularnet')
    parser.add_argument('data',
                        choices=['mnist', 'cifar10', 'cifar100', 'cerebrovascular'],
                        default='cerebrovascular')
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
    parser.add_argument('-dir',
                        '--dir-path',
                        type=str,
                        help='the directory to store the cerebrovascular dataset',
                        dest='dir_path')

    args = parser.parse_args()

    sgd = opt.SGD(lr=args.lr, momentum=0.9, weight_decay=1e-5, dtype=singa_dtype[args.precision])
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
        precision=args.precision,
        dir_path=args.dir_path)
