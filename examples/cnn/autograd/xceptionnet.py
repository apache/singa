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

from singa import autograd
from singa import tensor
from singa import device
from singa import layer
from singa import opt

import numpy as np
from tqdm import trange

# the code is modified from
# https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/xception.py


class Block(layer.Layer):

    def __init__(self,
                 in_filters,
                 out_filters,
                 reps,
                 strides=1,
                 padding=0,
                 start_with_relu=True,
                 grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = layer.Conv2d(in_filters,
                                     out_filters,
                                     1,
                                     stride=strides,
                                     padding=padding,
                                     bias=False)
            self.skipbn = layer.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.layers = []

        filters = in_filters
        if grow_first:
            self.layers.append(layer.ReLU())
            self.layers.append(
                layer.SeparableConv2d(in_filters,
                                      out_filters,
                                      3,
                                      stride=1,
                                      padding=1,
                                      bias=False))
            self.layers.append(layer.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps - 1):
            self.layers.append(layer.ReLU())
            self.layers.append(
                layer.SeparableConv2d(filters,
                                      filters,
                                      3,
                                      stride=1,
                                      padding=1,
                                      bias=False))
            self.layers.append(layer.BatchNorm2d(filters))

        if not grow_first:
            self.layers.append(layer.ReLU())
            self.layers.append(
                layer.SeparableConv2d(in_filters,
                                      out_filters,
                                      3,
                                      stride=1,
                                      padding=1,
                                      bias=False))
            self.layers.append(layer.BatchNorm2d(out_filters))

        if not start_with_relu:
            self.layers = self.layers[1:]
        else:
            self.layers[0] = layer.ReLU()

        if strides != 1:
            self.layers.append(layer.MaxPool2d(3, strides, padding + 1))

        self.register_layers(*self.layers)

        self.add = layer.Add()

    def forward(self, x):
        y = self.layers[0](x)
        for layer in self.layers[1:]:
            if isinstance(y, tuple):
                y = y[0]
            y = layer(y)

        if self.skip is not None:
            skip = self.skip(x)
            skip = self.skipbn(skip)
        else:
            skip = x
        y = self.add(y, skip)
        return y


__all__ = ['Xception']


class Xception(layer.Layer):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=1000):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        super(Xception, self).__init__()
        self.num_classes = num_classes

        self.conv1 = layer.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = layer.BatchNorm2d(32)
        self.relu1 = layer.ReLU()

        self.conv2 = layer.Conv2d(32, 64, 3, 1, 1, bias=False)
        self.bn2 = layer.BatchNorm2d(64)
        self.relu2 = layer.ReLU()
        # do relu here

        self.block1 = Block(64,
                            128,
                            2,
                            2,
                            padding=0,
                            start_with_relu=False,
                            grow_first=True)
        self.block2 = Block(128,
                            256,
                            2,
                            2,
                            padding=0,
                            start_with_relu=True,
                            grow_first=True)
        self.block3 = Block(256,
                            728,
                            2,
                            2,
                            padding=0,
                            start_with_relu=True,
                            grow_first=True)

        self.block4 = Block(728,
                            728,
                            3,
                            1,
                            start_with_relu=True,
                            grow_first=True)
        self.block5 = Block(728,
                            728,
                            3,
                            1,
                            start_with_relu=True,
                            grow_first=True)
        self.block6 = Block(728,
                            728,
                            3,
                            1,
                            start_with_relu=True,
                            grow_first=True)
        self.block7 = Block(728,
                            728,
                            3,
                            1,
                            start_with_relu=True,
                            grow_first=True)

        self.block8 = Block(728,
                            728,
                            3,
                            1,
                            start_with_relu=True,
                            grow_first=True)
        self.block9 = Block(728,
                            728,
                            3,
                            1,
                            start_with_relu=True,
                            grow_first=True)
        self.block10 = Block(728,
                             728,
                             3,
                             1,
                             start_with_relu=True,
                             grow_first=True)
        self.block11 = Block(728,
                             728,
                             3,
                             1,
                             start_with_relu=True,
                             grow_first=True)

        self.block12 = Block(728,
                             1024,
                             2,
                             2,
                             start_with_relu=True,
                             grow_first=False)

        self.conv3 = layer.SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = layer.BatchNorm2d(1536)
        self.relu3 = layer.ReLU()

        # Relu layer
        self.conv4 = layer.SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = layer.BatchNorm2d(2048)

        self.relu4 = layer.ReLU()
        self.globalpooling = layer.MaxPool2d(10, 1)
        self.flatten = layer.Flatten()
        self.fc = layer.Linear(2048, num_classes)

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = self.relu4(features)
        x = self.globalpooling(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


if __name__ == '__main__':
    model = Xception(num_classes=1000)
    print('Start intialization............')
    dev = device.create_cuda_gpu_on(0)
    #dev = device.create_cuda_gpu()

    niters = 20
    batch_size = 16
    IMG_SIZE = 299
    sgd = opt.SGD(lr=0.1, momentum=0.9, weight_decay=1e-5)

    tx = tensor.Tensor((batch_size, 3, IMG_SIZE, IMG_SIZE), dev)
    ty = tensor.Tensor((batch_size,), dev, tensor.int32)
    autograd.training = True
    x = np.random.randn(batch_size, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
    y = np.random.randint(0, 1000, batch_size, dtype=np.int32)
    tx.copy_from_numpy(x)
    ty.copy_from_numpy(y)

    with trange(niters) as t:
        for _ in t:
            x = model(tx)
            loss = autograd.softmax_cross_entropy(x, ty)
            sgd(loss)
