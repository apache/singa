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
from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import range
try:
    import pickle
except ImportError:
    import cPickle as pickle
import numpy as np
import os
import argparse
from tqdm import trange

from singa import optimizer
from singa import device
from singa import tensor
from caffe import caffe_net

import cnn
import vgg
import resnet


def load_dataset(filepath):
    print('Loading data file %s' % filepath)
    with open(filepath, 'rb') as fd:
        try:
            cifar10 = pickle.load(fd, encoding='latin1')
        except TypeError:
            cifar10 = pickle.load(fd)
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


def normalize_for_vgg(train_x, test_x):
    mean = train_x.mean()
    std = train_x.std()
    train_x -= mean
    test_x -= mean
    train_x /= std
    test_x /= std
    return train_x, test_x


def normalize_for_alexnet(train_x, test_x):
    mean = np.average(train_x, axis=0)
    train_x -= mean
    test_x -= mean
    return train_x, test_x


def vgg_lr(epoch):
    return 0.1 / float(1 << (epoch // 25))


def alexnet_lr(epoch):
    if epoch < 120:
        return 0.001
    elif epoch < 130:
        return 0.0001
    else:
        return 0.00001


def resnet_lr(epoch):
    if epoch < 81:
        return 0.1
    elif epoch < 122:
        return 0.01
    else:
        return 0.001


def caffe_lr(epoch):
    if epoch < 8:
        return 0.001
    else:
        return 0.0001


def train(data, net, max_epoch, get_lr, weight_decay, batch_size=100,
          use_cpu=False):
    print('Start intialization............')
    if use_cpu:
        print('Using CPU')
        dev = device.get_default_device()
    else:
        print('Using GPU')
        dev = device.create_cuda_gpu()

    net.to_device(dev)
    opt = optimizer.SGD(momentum=0.9, weight_decay=weight_decay)
    for (p, specs) in zip(net.param_names(), net.param_specs()):
        opt.register(p, specs)

    tx = tensor.Tensor((batch_size, 3, 32, 32), dev)
    ty = tensor.Tensor((batch_size,), dev, tensor.int32)
    train_x, train_y, test_x, test_y = data
    num_train_batch = train_x.shape[0] // batch_size
    num_test_batch = test_x.shape[0] // batch_size
    idx = np.arange(train_x.shape[0], dtype=np.int32)
    for epoch in range(max_epoch):
        np.random.shuffle(idx)
        loss, acc = 0.0, 0.0
        with trange(num_train_batch) as t:
            t.set_description('Epoch={}'.format(epoch))
            for b in t:
                x = train_x[idx[b * batch_size: (b + 1) * batch_size]]
                y = train_y[idx[b * batch_size: (b + 1) * batch_size]]
                tx.copy_from_numpy(x)
                ty.copy_from_numpy(y)
                grads, (l, a) = net.train(tx, ty)
                loss += l
                acc += a
                for (s, p, g) in zip(net.param_names(), net.param_values(), grads):
                    opt.apply_with_lr(epoch, get_lr(epoch), g, p, str(s), b)
                t.set_postfix(loss=l, accuracy=a)

        info = 'Training loss = %f, training accuracy = %f, lr = %f' \
            % ((loss / num_train_batch), (acc / num_train_batch), get_lr(epoch))
        print(info)

        loss, acc = 0.0, 0.0
        for b in range(num_test_batch):
            x = test_x[b * batch_size: (b + 1) * batch_size]
            y = test_y[b * batch_size: (b + 1) * batch_size]
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            l, a = net.evaluate(tx, ty)
            loss += l
            acc += a

        print('Test loss = %f, test accuracy = %f' %
              ((loss / num_test_batch), (acc / num_test_batch)))
    net.save('model', 20)  # save model params into checkpoint file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train dcnn for cifar10')
    parser.add_argument('model', choices=['vgg', 'cnn', 'resnet', 'caffe'],
                        default='vgg')
    parser.add_argument('data', default='cifar-10-batches-py')
    parser.add_argument('--use_cpu', action='store_true')
    args = parser.parse_args()
    assert os.path.exists(args.data), \
        'Pls download the cifar10 dataset via "download_data.py py"'
    print('Loading data ..................')
    train_x, train_y = load_train_data(args.data)
    test_x, test_y = load_test_data(args.data)
    if args.model == 'caffe':
        train_x, test_x = normalize_for_alexnet(train_x, test_x)
        net = caffe_net.create_net(args.use_cpu)
        # for cifar10_full_train_test.prototxt
        train((train_x, train_y, test_x, test_y), net, 160, alexnet_lr, 0.004,
              use_cpu=args.use_cpu)
        # for cifar10_quick_train_test.prototxt
        # train((train_x, train_y, test_x, test_y), net, 18, caffe_lr, 0.004,
        #      use_cpu=args.use_cpu)
    elif args.model == 'cnn':
        train_x, test_x = normalize_for_alexnet(train_x, test_x)
        net = cnn.create_net(args.use_cpu)
        train((train_x, train_y, test_x, test_y), net, 2, alexnet_lr, 0.004,
              use_cpu=args.use_cpu)
    elif args.model == 'vgg':
        train_x, test_x = normalize_for_vgg(train_x, test_x)
        net = vgg.create_net(args.use_cpu)
        train((train_x, train_y, test_x, test_y), net, 250, vgg_lr, 0.0005,
              use_cpu=args.use_cpu)
    else:
        train_x, test_x = normalize_for_alexnet(train_x, test_x)
        net = resnet.create_net(args.use_cpu)
        train((train_x, train_y, test_x, test_y), net, 200, resnet_lr, 1e-4,
              use_cpu=args.use_cpu)
