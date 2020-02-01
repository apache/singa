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
'''Predicting the labels for new images using the pre-trained alexnet model'''
from __future__ import print_function
from builtins import range
try:
    import pickle
except ImportError:
    import cPickle as pickle
import numpy as np

from singa import device
from singa import tensor
import alexnet


def predict(net, images, dev, topk=5):
    '''Predict the label of each image.

    Args:
        net, a pretrained neural net
        images, a batch of images [batch_size, 3, 32, 32], which have been
            pre-processed
        dev, the training device
        topk, return the topk labels for each image.
    '''
    x = tensor.from_numpy(images.astype(np.float32))
    x.to_device(dev)
    y = net.predict(x)
    y.to_host()
    prob = tensor.to_numpy(y)
    # prob = np.average(prob, 0)
    labels = np.flipud(np.argsort(prob))  # sort prob in descending order
    return labels[:, 0:topk]


def load_dataset(filepath):
    print('Loading data file %s' % filepath)
    with open(filepath, 'rb') as fd:
        cifar10 = pickle.load(fd, encoding='latin1')
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


def compute_image_mean(train_dir):
    images, _ = load_train_data(train_dir)
    return np.average(images, 0)

if __name__ == '__main__':
    model = alexnet.create_net(True)
    model.load('model', 20)  # the checkpoint from train.py
    dev = device.get_default_device()
    model.to_device(dev)

    mean = compute_image_mean('cifar-10-batches-py')
    test_images, _ = load_test_data('cifar-10-batches-py')
    # predict for two images
    print(predict(model, test_images[0:2] - mean, dev))
