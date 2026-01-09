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
