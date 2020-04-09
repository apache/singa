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
# under th

import os
import gzip
import numpy as np
import codecs

from singa import device
from singa import tensor
from singa import opt
from singa import autograd
from singa import sonnx
import onnx
from utils import check_exist_or_download

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')


def load_dataset():
    train_x_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
    train_y_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
    valid_x_url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
    valid_y_url = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
    train_x = read_image_file(check_exist_or_download(train_x_url)).astype(
        np.float32)
    train_y = read_label_file(check_exist_or_download(train_y_url)).astype(
        np.float32)
    valid_x = read_image_file(check_exist_or_download(valid_x_url)).astype(
        np.float32)
    valid_y = read_label_file(check_exist_or_download(valid_y_url)).astype(
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


def to_categorical(y, num_classes):
    y = np.array(y, dtype="int")
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    categorical = categorical.astype(np.float32)
    return categorical


class CNN:

    def __init__(self):
        self.conv1 = autograd.Conv2d(1, 20, 5, padding=0)
        self.conv2 = autograd.Conv2d(20, 50, 5, padding=0)
        self.linear1 = autograd.Linear(4 * 4 * 50, 500, bias=False)
        self.linear2 = autograd.Linear(500, 10, bias=False)
        self.pooling1 = autograd.MaxPool2d(2, 2, padding=0)
        self.pooling2 = autograd.MaxPool2d(2, 2, padding=0)

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


def accuracy(pred, target):
    y = np.argmax(pred, axis=1)
    t = np.argmax(target, axis=1)
    a = y == t
    return np.array(a, "int").sum() / float(len(t))


def train(model,
          x,
          y,
          epochs=1,
          batch_size=64,
          dev=device.get_default_device()):
    batch_number = x.shape[0] // batch_size

    for i in range(epochs):
        for b in range(batch_number):
            l_idx = b * batch_size
            r_idx = (b + 1) * batch_size

            x_batch = tensor.Tensor(device=dev, data=x[l_idx:r_idx])
            target_batch = tensor.Tensor(device=dev, data=y[l_idx:r_idx])

            output_batch = model.forward(x_batch)

            loss = autograd.softmax_cross_entropy(output_batch, target_batch)
            accuracy_rate = accuracy(tensor.to_numpy(output_batch),
                                     tensor.to_numpy(target_batch))

            sgd = opt.SGD(lr=0.001)
            for p, gp in autograd.backward(loss):
                sgd.update(p, gp)
            sgd.step()

            if b % 1e2 == 0:
                logging.info("acc %6.2f loss, %6.2f" %
                             (accuracy_rate, tensor.to_numpy(loss)[0]))
    logging.info("training completed")
    return x_batch, output_batch


def make_onnx(x, y):
    return sonnx.to_onnx([x], [y])


class Infer:

    def __init__(self, sg_ir):
        self.sg_ir = sg_ir
        for idx, tens in sg_ir.tensor_map.items():
            # allow the tensors to be updated
            tens.requires_grad = True
            tens.stores_grad = True

    def forward(self, x):
        return sg_ir.run([x])[0]


def re_train(sg_ir,
             x,
             y,
             epochs=1,
             batch_size=64,
             dev=device.get_default_device()):
    batch_number = x.shape[0] // batch_size

    new_model = Infer(sg_ir)

    for i in range(epochs):
        for b in range(batch_number):
            l_idx = b * batch_size
            r_idx = (b + 1) * batch_size

            x_batch = tensor.Tensor(device=dev, data=x[l_idx:r_idx])
            target_batch = tensor.Tensor(device=dev, data=y[l_idx:r_idx])

            output_batch = new_model.forward(x_batch)

            loss = autograd.softmax_cross_entropy(output_batch, target_batch)
            accuracy_rate = accuracy(tensor.to_numpy(output_batch),
                                     tensor.to_numpy(target_batch))

            sgd = opt.SGD(lr=0.01)
            for p, gp in autograd.backward(loss):
                sgd.update(p, gp)
            sgd.step()

            if b % 1e2 == 0:
                logging.info("acc %6.2f loss, %6.2f" %
                             (accuracy_rate, tensor.to_numpy(loss)[0]))
    logging.info("re-training completed")
    return new_model


class Trans:

    def __init__(self, sg_ir, last_layers):
        self.sg_ir = sg_ir
        self.last_layers = last_layers
        self.append_linear1 = autograd.Linear(500, 128, bias=False)
        self.append_linear2 = autograd.Linear(128, 32, bias=False)
        self.append_linear3 = autograd.Linear(32, 10, bias=False)

    def forward(self, x):
        y = sg_ir.run([x], last_layers=self.last_layers)[0]
        y = self.append_linear1(y)
        y = autograd.relu(y)
        y = self.append_linear2(y)
        y = autograd.relu(y)
        y = self.append_linear3(y)
        y = autograd.relu(y)
        return y


def transfer_learning(sg_ir,
                      x,
                      y,
                      epochs=1,
                      batch_size=64,
                      dev=device.get_default_device()):
    batch_number = x.shape[0] // batch_size

    trans_model = Trans(sg_ir, -1)

    for i in range(epochs):
        for b in range(batch_number):
            l_idx = b * batch_size
            r_idx = (b + 1) * batch_size

            x_batch = tensor.Tensor(device=dev, data=x[l_idx:r_idx])
            target_batch = tensor.Tensor(device=dev, data=y[l_idx:r_idx])
            output_batch = trans_model.forward(x_batch)

            loss = autograd.softmax_cross_entropy(output_batch, target_batch)
            accuracy_rate = accuracy(tensor.to_numpy(output_batch),
                                     tensor.to_numpy(target_batch))

            sgd = opt.SGD(lr=0.07)
            for p, gp in autograd.backward(loss):
                sgd.update(p, gp)
            sgd.step()

            if b % 1e2 == 0:
                logging.info("acc %6.2f loss, %6.2f" %
                             (accuracy_rate, tensor.to_numpy(loss)[0]))
    logging.info("transfer-learning completed")
    return trans_model


def test(model, x, y, batch_size=64, dev=device.get_default_device()):
    batch_number = x.shape[0] // batch_size

    result = 0
    for b in range(batch_number):
        l_idx = b * batch_size
        r_idx = (b + 1) * batch_size

        x_batch = tensor.Tensor(device=dev, data=x[l_idx:r_idx])
        target_batch = tensor.Tensor(device=dev, data=y[l_idx:r_idx])

        output_batch = model.forward(x_batch)
        result += accuracy(tensor.to_numpy(output_batch),
                           tensor.to_numpy(target_batch))

    logging.info("testing acc %6.2f" % (result / batch_number))


if __name__ == "__main__":
    # create device
    dev = device.create_cuda_gpu()
    #dev = device.get_default_device()
    # create model
    model = CNN()
    # load data
    train_x, train_y, valid_x, valid_y = load_dataset()
    # normalization
    train_x = train_x / 255
    valid_x = valid_x / 255
    train_y = to_categorical(train_y, 10)
    valid_y = to_categorical(valid_y, 10)
    # do training
    autograd.training = True
    x, y = train(model, train_x, train_y, dev=dev)
    onnx_model = make_onnx(x, y)
    # logging.info('The model is:\n{}'.format(onnx_model))

    # Save the ONNX model
    model_path = os.path.join('/', 'tmp', 'mnist.onnx')
    onnx.save(onnx_model, model_path)
    logging.info('The model is saved.')

    # load the ONNX model
    onnx_model = onnx.load(model_path)
    sg_ir = sonnx.prepare(onnx_model, device=dev)

    # inference
    autograd.training = False
    logging.info('The inference result is:')
    test(Infer(sg_ir), valid_x, valid_y, dev=dev)

    # re-training
    autograd.training = True
    new_model = re_train(sg_ir, train_x, train_y, dev=dev)
    autograd.training = False
    test(new_model, valid_x, valid_y, dev=dev)

    # transfer-learning
    autograd.training = True
    new_model = transfer_learning(sg_ir, train_x, train_y, dev=dev)
    autograd.training = False
    test(new_model, valid_x, valid_y, dev=dev)