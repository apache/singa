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
from singa import opt
from singa import tensor
import argparse
import numpy as np
import time
from PIL import Image

import sys
sys.path.append("../../..")

from healthcare.data import cifar10
from healthcare.models import tedct_net


def accuracy(pred, target):
    # y is network output to be compared with ground truth (int)
    y = np.argmax(pred, axis=1)
    a = y == target
    correct = np.array(a, "int").sum()
    return correct


def resize_dataset(x, image_size):
    num_data = x.shape[0]
    dim = x.shape[1]
    X = np.zeros(shape=(num_data, dim, image_size, image_size), dtype=np.float32)
    for n in range(0, num_data):
        for d in range(0, dim):
            X[n, d, :, :] = np.array(
                Image.fromarray(x[n, d, :, :]).resize(
                    (image_size, image_size), Image.BILINEAR
                ),
                dtype=np.float32,
            )
    return X


def run(
    local_rank,
    dir_path,
    max_epoch,
    batch_size,
    sgd,
    graph,
    verbosity,
    dist_option="plain",
    spars=None,
):
    # dev = device.create_cuda_gpu_on(local_rank)
    dev = device.get_default_device()
    dev.SetRandSeed(0)
    np.random.seed(0)

    train_x, train_y, val_x, val_y = cifar10.load(dir_path)

    num_channels = train_x.shape[1]
    data_size = np.prod(train_x.shape[1 : train_x.ndim]).item()
    num_classes = (np.max(train_y) + 1).item()

    backbone = tedct_net.create_cnn_model(num_channels=num_channels, num_classes=num_classes)
    model = tedct_net.create_model(backbone, prototype_count=10, lamb=0.5, temp=10)

    if backbone.dimension == 4:
        tx = tensor.Tensor(
            (batch_size, num_channels, backbone.input_size, backbone.input_size), dev
        )
        train_x = resize_dataset(train_x, backbone.input_size)
        val_x = resize_dataset(val_x, backbone.input_size)
    elif backbone.dimension == 2:
        tx = tensor.Tensor((batch_size, data_size), dev)
        np.reshape(train_x, (train_x.shape[0], -1))
        np.reshape(val_x, (val_x.shape[0], -1))

    ty = tensor.Tensor((batch_size,), dev, tensor.int32)
    num_train_batch = train_x.shape[0] // batch_size
    num_val_batch = val_x.shape[0] // batch_size
    idx = np.arange(train_x.shape[0], dtype=np.int32)

    model.set_optimizer(sgd)
    model.compile([tx], is_train=True, use_graph=graph, sequential=True)
    dev.SetVerbosity(verbosity)

    for epoch in range(max_epoch):
        print(f"Epoch {epoch}")
        np.random.shuffle(idx)

        train_correct = np.zeros(shape=[1], dtype=np.float32)
        test_correct = np.zeros(shape=[1], dtype=np.float32)
        train_loss = np.zeros(shape=[1], dtype=np.float32)

        model.train()
        for b in range(num_train_batch):
            x = train_x[idx[b * batch_size : (b + 1) * batch_size]]
            y = train_y[idx[b * batch_size : (b + 1) * batch_size]]
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)

            out, loss = model(tx, ty, dist_option, spars)
            train_correct += accuracy(tensor.to_numpy(out), y)
            train_loss += tensor.to_numpy(loss)[0]
        print(
            "Training loss = %f, training accuracy = %f"
            % (train_loss, train_correct / (num_train_batch * batch_size)),
            flush=True,
        )

    model.eval()
    for b in range(num_val_batch):
        x = val_x[b * batch_size : (b + 1) * batch_size]
        y = val_y[b * batch_size : (b + 1) * batch_size]

        tx.copy_from_numpy(x)
        ty.copy_from_numpy(y)

        out_test = model(tx, ty, dist_option="fp32", spars=None)
        test_correct += accuracy(tensor.to_numpy(out_test), y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a CPL model")
    parser.add_argument('-dir',
                        '--dir-path',
                        default="/tmp/cifar-10-batches-py",
                        type=str,
                        help='the directory to store the dataset',
                        dest='dir_path')
    parser.add_argument(
        "-m",
        "--max-epoch",
        default=20,
        type=int,
        help="maximum epochs",
        dest="max_epoch",
    )
    parser.add_argument(
        "-b", "--batch-size", default=64, type=int, help="batch size", dest="batch_size"
    )
    parser.add_argument(
        "-l",
        "--learning-rate",
        default=0.005,
        type=float,
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "-i",
        "--device-id",
        default=0,
        type=int,
        help="which GPU to use",
        dest="device_id",
    )
    parser.add_argument(
        "-g",
        "--disable-graph",
        default="True",
        action="store_false",
        help="disable graph",
        dest="graph",
    )
    parser.add_argument(
        "-v",
        "--log-verbosity",
        default=0,
        type=int,
        help="logging verbosity",
        dest="verbosity",
    )
    args = parser.parse_args()
    print(args)

    sgd = opt.SGD(lr=args.lr, momentum=0.9, weight_decay=1e-5)
    run(
        args.device_id,
        args.dir_path,
        args.max_epoch,
        args.batch_size,
        sgd,
        args.graph,
        args.verbosity
    )