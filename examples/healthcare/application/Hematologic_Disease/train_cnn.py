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

import time
from singa import singa_wrap as singa
from singa import device
from singa import tensor
from singa import opt
import numpy as np
from tqdm import tqdm
import argparse
import sys
sys.path.append("../../..")

from healthcare.data import bloodmnist
from healthcare.models import hematologic_net

np_dtype = {"float16": np.float16, "float32": np.float32}
singa_dtype = {"float16": tensor.float16, "float32": tensor.float32}


def accuracy(pred, target):
    """Compute recall accuracy.

    Args:
        pred (Numpy ndarray): Prediction array, should be in shape (B, C)
        target (Numpy ndarray): Ground truth array, should be in shape (B, ) 

    Return:
        correct (Float): Recall accuracy
    """
    # y is network output to be compared with ground truth (int)
    y = np.argmax(pred, axis=1)
    a = (y[:,None]==target).sum()
    correct = np.array(a, "int").sum()
    return correct

def run(dir_path,
        max_epoch,
        batch_size,
        model,
        data,
        lr,
        graph,
        verbosity,
        dist_option='plain',
        spars=None,
        precision='float32'):
    # Start training
    dev = device.create_cpu_device()
    dev.SetRandSeed(0)
    np.random.seed(0)
    if data == 'bloodmnist':
        train_dataset, val_dataset, num_class = bloodmnist.load(dir_path=dir_path)
    else:
        print(
            'Wrong dataset!'
        )
        sys.exit(0)

    if model == 'cnn':
        model = hematologic_net.create_model(num_classes=num_class)
    else:
        print(
            'Wrong model!'
        )
        sys.exit(0)

    # Model configuration for CNN
    # criterion = layer.SoftMaxCrossEntropy()
    optimizer_ft = opt.Adam(lr)

    tx = tensor.Tensor(
        (batch_size, 3, model.input_size, model.input_size), dev,
        singa_dtype[precision])
    ty = tensor.Tensor((batch_size,), dev, tensor.int32)

    num_train_batch = train_dataset.__len__() // batch_size
    num_val_batch = val_dataset.__len__() // batch_size
    idx = np.arange(train_dataset.__len__(), dtype=np.int32)

    # Attach model to graph
    model.set_optimizer(optimizer_ft)
    model.compile([tx], is_train=True, use_graph=graph, sequential=False)
    dev.SetVerbosity(verbosity)

    # Training and evaluation loop
    for epoch in range(max_epoch):
        print(f'Epoch {epoch}:')

        start_time = time.time()

        train_correct = np.zeros(shape=[1], dtype=np.float32)
        test_correct = np.zeros(shape=[1], dtype=np.float32)
        train_loss = np.zeros(shape=[1], dtype=np.float32)

        # Training part
        model.train()
        for b in tqdm(range(num_train_batch)):
            # Extract batch from image list
            x, y = train_dataset.batchgenerator(idx[b * batch_size:(b + 1) * batch_size],
                batch_size=batch_size, data_size=(3, model.input_size, model.input_size))
            x = x.astype(np_dtype[precision])

            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)

            out, loss = model(tx, ty, dist_option, spars)
            train_correct += accuracy(tensor.to_numpy(out), y)
            train_loss += tensor.to_numpy(loss)[0]
        print('Training loss = %f, training accuracy = %f' %
                      (train_loss, train_correct /
                       (num_train_batch * batch_size)))

        # Validation part
        model.eval()
        for b in tqdm(range(num_val_batch)):
            x, y = train_dataset.batchgenerator(idx[b * batch_size:(b + 1) * batch_size],
                batch_size=batch_size, data_size=(3, model.input_size, model.input_size))
            x = x.astype(np_dtype[precision])

            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)

            out = model(tx)
            test_correct += accuracy(tensor.to_numpy(out), y)

        print('Evaluation accuracy = %f, Elapsed Time = %fs' %
                      (test_correct / (num_val_batch * batch_size),
                       time.time() - start_time))


if __name__ == '__main__':
    # Use argparse to get command config: max_epoch, model, data, etc., for single gpu training
    parser = argparse.ArgumentParser(
        description='Training using the autograd and graph.')
    parser.add_argument(
        'model',
        choices=['cnn'],
        default='cnn')
    parser.add_argument('data',
                        choices=['bloodmnist'],
                        default='bloodmnist')
    parser.add_argument('-p',
                        choices=['float32', 'float16'],
                        default='float32',
                        dest='precision')
    parser.add_argument('-dir',
                        '--dir-path',
                        default="/tmp/bloodmnist",
                        type=str,
                        help='the directory to store the bloodmnist dataset',
                        dest='dir_path')
    parser.add_argument('-m',
                        '--max-epoch',
                        default=100,
                        type=int,
                        help='maximum epochs',
                        dest='max_epoch')
    parser.add_argument('-b',
                        '--batch-size',
                        default=256,
                        type=int,
                        help='batch size',
                        dest='batch_size')
    parser.add_argument('-l',
                        '--learning-rate',
                        default=0.003,
                        type=float,
                        help='initial learning rate',
                        dest='lr')
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

    run(args.dir_path,
        args.max_epoch,
        args.batch_size,
        args.model,
        args.data,
        args.lr,
        args.graph,
        args.verbosity,
        precision=args.precision)
