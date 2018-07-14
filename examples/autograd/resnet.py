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

# the code is modified from
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py

from singa import autograd
from singa import tensor
from singa import device
from singa import utils
from singa import optimizer

import numpy as np


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return autograd.Conv2D(in_planes, out_planes, kernel_size=3, stride=stride,
                           padding=1, bias=False)


class BasicBlock(autograd.Layer):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = autograd.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = autograd.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def __call__(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = autograd.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = autograd.add(out, residual)
        out = autograd.relu(out)

        return out


class Bottleneck(autograd.Layer):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = autograd.Conv2D(
            inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = autograd.BatchNorm2d(planes)
        self.conv2 = autograd.Conv2D(planes, planes, kernel_size=3, stride=stride,
                                     padding=1, bias=False)
        self.bn2 = autograd.BatchNorm2d(planes)
        self.conv3 = autograd.Conv2D(
            planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = autograd.BatchNorm2d(planes * self.expansion)

        self.downsample = downsample
        self.stride = stride

    def __call__(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = autograd.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = autograd.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = autograd.add(out, residual)
        out = autograd.relu(out)

        return out


class ResNet(autograd.Layer):

    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = autograd.Conv2D(3, 64, kernel_size=7, stride=2, padding=3,
                                     bias=False)
        self.bn1 = autograd.BatchNorm2d(64)
        self.maxpool = autograd.MaxPool2d(
            kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = autograd.AvgPool2d(7, stride=1)
        self.fc = autograd.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            conv = autograd.Conv2D(self.inplanes, planes * block.expansion,
                                   kernel_size=1, stride=stride, bias=False)
            bn = autograd.BatchNorm2d(planes * block.expansion),
            downsample = lambda x: bn(conv(x))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        def forward(x):
            for layer in layers:
                x = layer(x)
            return x
        return forward

    def __call__(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = autograd.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = autograd.flatten(x)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)

    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)

    return model


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


def accuracy(pred, target):
    y = np.argmax(pred, axis=1)
    t = np.argmax(target, axis=1)
    a = y == t
    return np.array(a, 'int').sum() / float(len(t))


def train(data, net, max_epoch, get_lr, weight_decay=1e-5, batch_size=100):
    print('Start intialization............')
    dev = device.create_cuda_gpu()

    opt = optimizer.SGD(momentum=0.9, weight_decay=weight_decay)

    tx = tensor.Tensor((batch_size, 3, 32, 32), dev)
    ty = tensor.Tensor((batch_size,), dev, tensor.int32)
    train_x, train_y, test_x, test_y = data
    num_train_batch = train_x.shape[0] // batch_size
    num_test_batch = test_x.shape[0] // batch_size
    idx = np.arange(train_x.shape[0], dtype=np.int32)
    for epoch in range(max_epoch):
        np.random.shuffle(idx)
        loss, acc = 0.0, 0.0
        print('Epoch %d' % epoch)
        autograd.training = True
        for b in range(num_train_batch):
            x = train_x[idx[b * batch_size: (b + 1) * batch_size]]
            y = train_y[idx[b * batch_size: (b + 1) * batch_size]]
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            x = net(tx)
            loss = autograd.softmax_cross_entropy(x, ty)
            np_loss = tensor.to_numpy(loss)
            acc += accuracy(tensor.to_numpy(x), y)

            for p, g in autograd.backwards(loss):
                opt.apply_with_lr(epoch, get_lr(epoch), g, p)
            # update progress bar
            utils.update_progress(b * 1.0 / num_train_batch,
                                  'training loss = %f' % (np_loss[0]))

        loss, acc = 0.0, 0.0
        autograd.training = True
        for b in range(num_test_batch):
            x = test_x[b * batch_size: (b + 1) * batch_size]
            y = test_y[b * batch_size: (b + 1) * batch_size]
            tx.copy_from_numpy(x)
            ty.copy_from_numpy(y)
            x = net(tx)
            l = autograd.softmax_cross_entropy(x, ty)
            loss += tensor.to_numpy(l)[0]
            acc += accuracy(x, y)

        print('test loss = %f, test accuracy = %f' %
              ((loss / num_test_batch), (acc / num_test_batch)))


def resnet_lr(epoch):
    if epoch < 81:
        return 0.1
    elif epoch < 122:
        return 0.01
    else:
        return 0.001


if __name__ == '__main__':
    model = resnet18()
    train_x, train_y = load_train_data()
    test_x, test_y = load_test_data()
    mean = np.average(train_x, axis=0)
    train_x -= mean
    test_x -= mean
    train(model, (train_x, train_y, test_x, test_y), 10, resnet_lr)
