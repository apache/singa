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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
""" CIFAR10 dataset is at https://www.cs.toronto.edu/~kriz/cifar.html.
It includes 5 binary dataset, each contains 10000 images. 1 row (1 image)
includes 1 label & 3072 pixels.  3072 pixels are 3 channels of a 32x32 image
"""

import cPickle
import numpy as np
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '../../build/python'))
from singa import initializer
from singa import utils
from singa import optimizer
from singa import device
from singa import tensor
from singa.proto import core_pb2

import alexnet


def load_dataset(filepath):
    print 'Loading data file %s' % filepath
    with open(filepath, 'rb') as fd:
        cifar10 = cPickle.load(fd)
    image = cifar10['data'].astype(dtype=np.uint8)
    image = image.reshape((-1, 3, 32, 32))
    label = np.asarray(cifar10['labels'], dtype=np.uint8)
    label = label.reshape(label.size, 1)
    return image, label


def load_train_data(dir_path, num_batches=5):
    labels = []
    batchsize = 10000
    images = np.empty((num_batches * batchsize, 3, 32, 32), dtype=np.uint8)
    for did in range(1, num_batches + 1):
        fname_train_data = dir_path + "/data_batch_{}".format(did)
        image, label = load_dataset(fname_train_data)
        images[(did - 1) * batchsize:did * batchsize] = image
        labels.extend(label)
    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    return images, labels


def load_test_data(dir_path):
    images, labels = load_dataset(dir_path + "/test_batch")
    return np.array(images,  dtype=np.float32), np.array(labels, dtype=np.int32)


def get_lr(epoch):
    if epoch < 120:
        return 0.001
    elif epoch < 130:
        return 0.0001
    elif epoch < 140:
        return 0.00001


def train(data_dir, net, num_epoch=140, batch_size=100):
    print 'Start intialization............'
    cuda = device.create_cuda_gpu()
    net.to_device(cuda)
    opt = optimizer.SGD(momentum=0.9, weight_decay=0.004)
    for (p, specs) in zip(net.param_values(), net.param_specs()):
        filler = specs.filler
        if filler.type == 'gaussian':
            initializer.gaussian(p, filler.mean, filler.std)
        else:
            p.set_value(0)
        opt.register(p, specs)
        print specs.name, filler.type, p.l1()
    print 'Loading data ..................'
    train_x, train_y = load_train_data(data_dir)
    test_x, test_y = load_test_data(data_dir)
    mean = np.average(train_x, axis=0)
    train_x -= mean
    test_x -= mean

    tx = tensor.Tensor((batch_size, 3, 32, 32), cuda)
    ty = tensor.Tensor((batch_size,), cuda, core_pb2.kInt)
    num_train_batch = train_x.shape[0] / batch_size
    num_test_batch = test_x.shape[0] / batch_size
    idx = np.arange(train_x.shape[0], dtype=np.int32)
    for epoch in range(num_epoch):
        np.random.shuffle(idx)
        loss, acc = 0.0, 0.0
        print 'Epoch %d' % epoch
        for b in range(num_train_batch):
            x = train_x[idx[b * batch_size: (b + 1) * batch_size]]
            y = train_y[idx[b * batch_size: (b + 1) * batch_size]]
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            grads, (l, a) = net.train(tx, ty)
            loss += l
            acc += a
            for (s, p, g) in zip(net.param_specs(), net.param_values(), grads):
                opt.apply_with_lr(epoch, get_lr(epoch), g, p, str(s.name))
            # update progress bar
            utils.update_progress(b * 1.0 / num_train_batch,
                                  'training loss = %f, accuracy = %f' % (l, a))
        info = 'training loss = %f, training accuracy = %f' \
            % (loss / num_train_batch, acc / num_train_batch)
        print info

        loss, acc = 0.0, 0.0
        for b in range(num_test_batch):
            x = test_x[b * batch_size: (b + 1) * batch_size]
            y = test_y[b * batch_size: (b + 1) * batch_size]
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            l, a = net.evaluate(tx, ty)
            loss += l
            acc += a

        print 'test loss = %f, test accuracy = %f' \
            % (loss / num_test_batch, acc / num_test_batch)
    net.save('model.bin')  # save model params into checkpoint file

if __name__ == '__main__':
    data_dir = 'cifar-10-batches-py'
    assert os.path.exists(data_dir), \
        'Pls download the cifar10 dataset via "download_data.py py"'
    net = alexnet.create_net()
    train(data_dir, net)
