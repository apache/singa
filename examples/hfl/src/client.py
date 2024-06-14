#!/usr/bin/env python3
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

# modified from https://github.com/apache/singa/blob/master/examples/cnn/train_cnn.py
# modified from https://github.com/zhengzangw/Fed-SINGA/blob/main/src/client/app.py
# modified from https://github.com/zhengzangw/Fed-SINGA/blob/main/src/client/main.py

import socket

from .proto import interface_pb2 as proto
from .proto import utils
from .proto.utils import parseargs

import time
import numpy as np
from PIL import Image
from singa import device, opt, tensor
from tqdm import tqdm

from . import bank
from . import mlp

np_dtype = {"float16": np.float16, "float32": np.float32}
singa_dtype = {"float16": tensor.float16, "float32": tensor.float32}


class Client:
    """Client sends and receives protobuf messages.

    Create and start the server, then use pull and push to communicate with the server.

    Attributes:
        global_rank (int): The rank in training process.
        host (str): Host address of the server.
        port (str): Port of the server.
        sock (socket.socket): Socket of the client.
        weights (Dict[Any]): Weights stored locally.
    """

    def __init__(
        self,
        global_rank: int = 0,
        host: str = "127.0.0.1",
        port: str = 1234,
    ) -> None:
        """Class init method

        Args:
            global_rank (int, optional): The rank in training process. Defaults to 0.
                Provided by the '-i' parameter (device_id) in the running script.
            host (str, optional): Host ip address. Defaults to '127.0.0.1'.
            port (str, optional): Port. Defaults to 1234.
        """
        self.host = host
        self.port = port
        self.global_rank = global_rank

        self.sock = socket.socket()

        self.weights = {}

    def __start_connection(self) -> None:
        """Start the network connection to server."""
        self.sock.connect((self.host, self.port))

    def __start_rank_pairing(self) -> None:
        """Sending global rank to server"""
        utils.send_int(self.sock, self.global_rank)

    def start(self) -> None:
        """Start the client.

        This method will first connect to the server. Then global rank is sent to the server.
        """
        self.__start_connection()
        self.__start_rank_pairing()

        print(f"[Client {self.global_rank}] Connect to {self.host}:{self.port}")

    def close(self) -> None:
        """Close the server."""
        self.sock.close()

    def pull(self) -> None:
        """Client pull weights from server.

        Namely server push weights from clients.
        """
        message = proto.WeightsExchange()
        message = utils.receive_message(self.sock, message)
        for k, v in message.weights.items():
            self.weights[k] = utils.deserialize_tensor(v)

    def push(self) -> None:
        """Client push weights to server.

        Namely server pull weights from clients.
        """
        message = proto.WeightsExchange()
        message.op_type = proto.GATHER
        for k, v in self.weights.items():
            message.weights[k] = utils.serialize_tensor(v)
        utils.send_message(self.sock, message)


# Data augmentation
def augmentation(x, batch_size):
    xpad = np.pad(x, [[0, 0], [0, 0], [4, 4], [4, 4]], "symmetric")
    for data_num in range(0, batch_size):
        offset = np.random.randint(8, size=2)
        x[data_num, :, :, :] = xpad[data_num, :,
                                    offset[0]:offset[0] + x.shape[2],
                                    offset[1]:offset[1] + x.shape[2]]
        if_flip = np.random.randint(2)
        if if_flip:
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
            X[n, d, :, :] = np.array(
                Image.fromarray(x[n, d, :, :]).resize((image_size, image_size),
                                                      Image.BILINEAR),
                dtype=np.float32,
            )
    return X


def get_data(data, data_dist="iid", device_id=None):
    if data == "bank":
        train_x, train_y, val_x, val_y, num_classes = bank.load(device_id)
    else:
        raise NotImplementedError
    return train_x, train_y, val_x, val_y, num_classes


def get_model(model, num_channels=None, num_classes=None, data_size=None):
    if model == "mlp":
        model = mlp.create_model(data_size=data_size, num_classes=num_classes)
    else:
        raise NotImplementedError
    return model


def run(
    global_rank,
    world_size,
    device_id,
    max_epoch,
    batch_size,
    model,
    data,
    data_dist,
    sgd,
    graph,
    verbosity,
    dist_option="plain",
    spars=None,
    precision="float32",
):
    # Connect to server
    client = Client(global_rank=device_id)
    client.start()

    dev = device.get_default_device()
    dev.SetRandSeed(0)
    np.random.seed(0)

    # Prepare dataset
    train_x, train_y, val_x, val_y, num_classes = get_data(
        data, data_dist, device_id)

    num_channels = train_x.shape[1]
    data_size = np.prod(train_x.shape[1:train_x.ndim]).item()

    # Prepare model
    model = get_model(model,
                      num_channels=num_channels,
                      num_classes=num_classes,
                      data_size=data_size)

    if model.dimension == 4:
        image_size = train_x.shape[2]

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
            (batch_size, num_channels, model.input_size, model.input_size),
            dev,
            singa_dtype[precision],
        )
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
        if epoch > 0:
            client.pull()
            model.set_states(client.weights)

        if global_rank == 0:
            print("Starting Epoch %d:" % (epoch))

        start_time = time.time()
        np.random.shuffle(idx)

        # Training phase
        max_inner_epoch = 1
        for inner_epoch in range(max_inner_epoch):
            train_correct = np.zeros(shape=[1], dtype=np.float32)
            train_loss = np.zeros(shape=[1], dtype=np.float32)
            test_correct = np.zeros(shape=[1], dtype=np.float32)

            model.train()
            for b in tqdm(range(num_train_batch)):
                # Generate the patch data in this iteration
                x = train_x[idx[b * batch_size:(b + 1) * batch_size]]
                if model.dimension == 4:
                    x = augmentation(x, batch_size)
                    if image_size != model.input_size:
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
                train_acc = train_correct / (num_train_batch * batch_size *
                                             world_size)
                print(
                    "[inner epoch %d] Training loss = %f, training accuracy = %f"
                    % (inner_epoch, train_loss, train_acc),
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
                # Reduce the evaluation accuracy from multiple devices
                test_correct = reduce_variable(test_correct, sgd, reducer)

            # Output the evaluation accuracy
            if global_rank == 0:
                print(
                    '[inner epoch %d] Evaluation accuracy = %f, Elapsed Time = %fs'
                    % (inner_epoch, test_correct /
                       (num_val_batch * batch_size * world_size),
                       time.time() - start_time),
                    flush=True)

        client.weights = model.get_states()
        client.push()

    dev.PrintTimeProfiling()

    client.close()


if __name__ == "__main__":
    args = parseargs()
    sgd = opt.SGD(lr=args.lr,
                  momentum=0.9,
                  weight_decay=1e-5,
                  dtype=singa_dtype[args.precision])
    run(
        0,
        1,
        args.device_id,
        args.max_epoch,
        args.batch_size,
        args.model,
        args.data,
        args.data_dist,
        sgd,
        args.graph,
        args.verbosity,
        precision=args.precision,
    )
