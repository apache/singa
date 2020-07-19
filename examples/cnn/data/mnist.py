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

import numpy as np
import os
import sys
import gzip
import codecs


def check_dataset_exist(dirpath):
    if not os.path.exists(dirpath):
        print(
            'The MNIST dataset does not exist. Please download the mnist dataset using python data/download_mnist.py'
        )
        sys.exit(0)
    return dirpath


def load_dataset():
    train_x_path = '/tmp/train-images-idx3-ubyte.gz'
    train_y_path = '/tmp/train-labels-idx1-ubyte.gz'
    valid_x_path = '/tmp/t10k-images-idx3-ubyte.gz'
    valid_y_path = '/tmp/t10k-labels-idx1-ubyte.gz'

    train_x = read_image_file(check_dataset_exist(train_x_path)).astype(
        np.float32)
    train_y = read_label_file(check_dataset_exist(train_y_path)).astype(
        np.float32)
    valid_x = read_image_file(check_dataset_exist(valid_x_path)).astype(
        np.float32)
    valid_y = read_label_file(check_dataset_exist(valid_y_path)).astype(
        np.float32)
    return train_x, train_y, valid_x, valid_y


def read_label_file(path):
    with gzip.open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2049
        length = get_int(data[4:8])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=8).reshape((length))
        return parsed


def get_int(b):
    return int(codecs.encode(b, 'hex'), 16)


def read_image_file(path):
    with gzip.open(path, 'rb') as f:
        data = f.read()
        assert get_int(data[:4]) == 2051
        length = get_int(data[4:8])
        num_rows = get_int(data[8:12])
        num_cols = get_int(data[12:16])
        parsed = np.frombuffer(data, dtype=np.uint8, offset=16).reshape(
            (length, 1, num_rows, num_cols))
        return parsed


def normalize(train_x, val_x):
    train_x /= 255
    val_x /= 255
    return train_x, val_x


def load():
    train_x, train_y, val_x, val_y = load_dataset()
    train_x, val_x = normalize(train_x, val_x)
    train_x = train_x.astype(np.float32)
    val_x = val_x.astype(np.float32)
    train_y = train_y.astype(np.int32)
    val_y = val_y.astype(np.int32)
    return train_x, train_y, val_x, val_y
