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

from singa import device
from singa import tensor
from singa import autograd
from singa import opt
from singa import sonnx
import onnx

import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def load_data(path):
    f = np.load(path)
    x_train, y_train = f['x_train'], f['y_train']
    x_test, y_test = f['x_test'], f['y_test']
    f.close()
    return (x_train, y_train), (x_test, y_test)


def to_categorical(y, num_classes):
    '''
    Converts a class vector (integers) to binary class matrix.
    Args
        y: class vector to be converted into a matrix
            (integers from 0 to num_classes).
        num_classes: total number of classes.
    Return
        A binary matrix representation of the input.
    '''
    y = np.array(y, dtype='int')
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    categorical = categorical.astype(np.float32)
    return categorical


def preprocess(data):
    data = data.astype(np.float32)
    data /= 255
    data = np.expand_dims(data, axis=1)
    return data


def accuracy(pred, target):
    y = np.argmax(pred, axis=1)
    t = np.argmax(target, axis=1)
    a = y == t
    return np.array(a, 'int').sum() / float(len(t))


if __name__ == '__main__':


    file_path = 'mnist.npz'

    print(file_path)
    assert os.path.exists(file_path), \
        'Pls download the MNIST dataset from https://s3.amazonaws.com/img-datasets/mnist.npz'
    use_cpu=False
    if use_cpu:
        print('Using CPU')
        dev = device.get_default_device()
    else:
        print('Using GPU')
        dev = device.create_cuda_gpu()

    train, test = load_data(file_path)
    print(train[0].shape)
    batch_number = 600
    num_classes = 10
    epochs = 1

    sgd = opt.SGD(lr=0.00)

    x_train = preprocess(train[0])
    y_train = to_categorical(train[1], num_classes)

    x_test = preprocess(test[0])
    y_test = to_categorical(test[1], num_classes)
    print('the shape of training data is', x_train.shape)
    print('the shape of training label is', y_train.shape)
    print('the shape of testing data is', x_test.shape)
    print('the shape of testing label is', y_test.shape)


    model = onnx.load('cnn.onnx')
    rep = sonnx.BackendRep(model,dev)
    #####backend run multiple times
    print('finish init')
    autograd.training = True
    # training process
    for epoch in range(1):
        inputs = tensor.Tensor(device=dev, data=x_train[0:100], stores_grad=False)
        targets = tensor.Tensor(device=dev, data=y_train[0:100], requires_grad=False, stores_grad=False)
        y0 = rep.run([inputs])[0]
        loss = autograd.softmax_cross_entropy(y0,targets)
        print('outputs',tensor.to_numpy(loss)[0])

#####backend run only one time
y0 = sonnx.Backend.run_model(model,[inputs],dev)[0]
loss = autograd.softmax_cross_entropy(y0, targets)
print('training loss = ', tensor.to_numpy(loss)[0])