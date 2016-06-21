#!/usr/bin/env python

# /************************************************************
# *
# * Licensed to the Apache Software Foundation (ASF) under one
# * or more contributor license agreements.  See the NOTICE file
# * distributed with this work for additional information
# * regarding copyright ownership.  The ASF licenses this file
# * to you under the Apache License, Version 2.0 (the
# * "License"); you may not use this file except in compliance
# * with the License.  You may obtain a copy of the License at
# *
# *   http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing,
# * software distributed under the License is distributed on an
# * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# * KIND, either express or implied.  See the License for the
# * specific language governing permissions and limitations
# * under the License.
# *
# *************************************************************/

import sys
import os
import numpy as np

from . import singa_wrap as singa

from .proto.core_pb2 import *
from .proto.model_pb2 import *


class Layer(object):

    def __init__(self, name='default', **kwargs):
        self.singa_layer = singa.Layer()
        self.conf = LayerConf()
        self.conf.name = name
        # other initialization
        # ...

    def setup(self, proto):
        self.singa_layer.Setup(proto.SerializeToString())

    def forward(self, flag, inputs):
        return self.singa_layer.Forward(flag, inputs)

    def backward(self, flag, grads):
        return self.singa_layer.Backward(flag, grads)

    def to_device(self, device):
        self.singa_layer.ToDevice(device)

    def as_type(self, dtype):
        self.singa_layer.AsType(dtype)

    def name(self):
        return self.singa_layer.name()


class Conv2D(Layer):

    def __init__(self, in_channels, out_channels, kernel=3, stride=1,
                 border_mode='valid',  engine='cudnn', cudnn_prefer='fatest',
                 data_format='NCHW', use_bias=True, pad=None, W_specs=None,
                 b_specs=None, name=None):

        super(Conv2D, self).__init__(name)

        conf = ConvolutionConf()
        conf.channels = in_channels
        conf.num_output = out_channels
        # other fields
        # ...

        self.conf.convolution_conf.CopyFrom(conf)

        self.setup(self.conf)
