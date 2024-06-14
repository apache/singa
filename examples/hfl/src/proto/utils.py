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

import argparse
import pickle
import socket
import struct

from google.protobuf.message import Message
from singa import tensor


def receive_all(conn: socket.socket, size: int) -> bytes:
    """Receive a given length of bytes from socket.

    Args:
        conn (socket.socket): Socket connection.
        size (int): Length of bytes to receive.

    Raises:
        RuntimeError: If connection closed before chunk was read, it will raise an error.

    Returns:
        bytes: Received bytes.
    """
    buffer = b""
    while size > 0:
        chunk = conn.recv(size)
        if not chunk:
            raise RuntimeError("connection closed before chunk was read")
        buffer += chunk
        size -= len(chunk)
    return buffer


def send_int(conn: socket.socket, i: int, pack_format: str = "Q") -> None:
    """Send an integer from socket.

    Args:
        conn (socket.socket): Socket connection.
        i (int): Integer to send.
        pack_format (str, optional): Pack format. Defaults to "Q", which means unsigned long long.
    """
    data = struct.pack(f"!{pack_format}", i)
    conn.sendall(data)


def receive_int(conn: socket.socket, pack_format: str = "Q") -> int:
    """Receive an integer from socket.

    Args:
        conn (socket.socket): Socket connection.
        pack_format (str, optional): Pack format. Defaults to "Q", which means unsigned long long.

    Returns:
        int: Received integer.
    """
    buffer_size = struct.Struct(pack_format).size
    data = receive_all(conn, buffer_size)
    (data,) = struct.unpack(f"!{pack_format}", data)
    return data


def send_message(conn: socket.socket,
                 data: Message,
                 pack_format: str = "Q") -> None:
    """Send protobuf message from socket. First the length of protobuf message will be sent. Then the message is sent.

    Args:
        conn (socket.socket): Socket connection.
        data (Message): Protobuf message to send.
        pack_format (str, optional): Length of protobuf message pack format. Defaults to "Q", which means unsigned long long.
    """
    send_int(conn, data.ByteSize(), pack_format)
    conn.sendall(data.SerializePartialToString())


def receive_message(conn: socket.socket,
                    data: Message,
                    pack_format: str = "Q") -> Message:
    """Receive protobuf message from socket

    Args:
        conn (socket.socket): Socket connection.
        data (Message): Placehold for protobuf message.
        pack_format (str, optional): Length of protobuf message pack format. Defaults to "Q", which means unsigned long long.

    Returns:
        Message: The protobuf message.
    """
    data_len = receive_int(conn, pack_format)
    data.ParseFromString(receive_all(conn, data_len))
    return data


def serialize_tensor(t: tensor.Tensor) -> bytes:
    """Serialize a singa tensor to bytes.

    Args:
        t (tensor.Tensor): The singa tensor.

    Returns:
        bytes: The serialized tensor.
    """
    return pickle.dumps(tensor.to_numpy(t), protocol=0)


def deserialize_tensor(t: bytes) -> tensor.Tensor:
    """Recover singa tensor from bytes.

    Args:
        t (bytes): The serialized tensor.

    Returns:
        tensor.Tensor: The singa tensor.
    """
    return tensor.from_numpy(pickle.loads(t))


def parseargs(arg=None) -> argparse.Namespace:
    """Parse command line arguments

    Returns:
        argparse.Namespace: parsed arguments
    """

    parser = argparse.ArgumentParser(
        description="Training using the autograd and graph.")
    parser.add_argument(
        "--model",
        choices=["cnn", "resnet", "xceptionnet", "mlp", "alexnet"],
        default="mlp")
    parser.add_argument("--data",
                        choices=["mnist", "cifar10", "cifar100", "bank"],
                        default="mnist")
    parser.add_argument("-p",
                        choices=["float32", "float16"],
                        default="float32",
                        dest="precision")
    parser.add_argument("-m",
                        "--max-epoch",
                        default=10,
                        type=int,
                        help="maximum epochs",
                        dest="max_epoch")
    parser.add_argument("-b",
                        "--batch-size",
                        default=64,
                        type=int,
                        help="batch size",
                        dest="batch_size")
    parser.add_argument("-l",
                        "--learning-rate",
                        default=0.005,
                        type=float,
                        help="initial learning rate",
                        dest="lr")
    # Determine which gpu to use
    parser.add_argument("-i",
                        "--device-id",
                        default=0,
                        type=int,
                        help="which GPU to use",
                        dest="device_id")
    parser.add_argument(
        "-g",
        "--disable-graph",
        default="True",
        action="store_false",
        help="disable graph",
        dest="graph",
    )
    parser.add_argument("-v",
                        "--log-verbosity",
                        default=0,
                        type=int,
                        help="logging verbosity",
                        dest="verbosity")
    parser.add_argument(
        "-d",
        "--data-distribution",
        choices=["iid", "non-iid"],
        default="iid",
        help="data distribution",
        dest="data_dist",
    )
    parser.add_argument("--num_clients", default=10, type=int)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=1234)

    args = parser.parse_args(arg)
    return args
