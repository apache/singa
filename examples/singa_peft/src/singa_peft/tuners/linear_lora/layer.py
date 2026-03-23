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

import math
from singa import tensor
from singa import autograd
from singa import layer


class LinearLoRALayer(layer.Layer):
    """
    LinearLoRALayer: LoRA implemented in a linear layer
    """
    def __init__(
            self,
            base_layer: layer.Linear,
            r: int = 8,
            lora_alpha: int = 1,
            lora_dropout: float = 0.,
    ):
        r"""
        Args:
            base_layer: a linear layer, The input and output channels of the linear lora layer are equal to this base layer.
            r: the rank in LoRA, which determines the size of the low-rank matrix. An integer greater than 0 is required, default 8.
            lora_alpha: learning rate scaling factor, default 1
            lora_dropout: dropout ratio, default 0.
        """
        super().__init__()
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        self.r = r
        self.base_layer = base_layer
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.merged = False


    def initialize(self, x):
        # freeze weights of base layer
        if self.base_layer._initialized is False:
            self.base_layer.initialize(x)
        self.freeze_pretrained_weight(True)
        # actual trainable parameters
        lora_A_shape = (self.r, self.in_features)
        lora_B_shape = (self.out_features, self.r)
        self.lora_A = tensor.Tensor(
            shape=lora_A_shape,
            dtype=x.dtype,
            requires_grad=True,
            stores_grad=True
        )
        self.lora_B = tensor.Tensor(
            shape=lora_B_shape,
            dtype=x.dtype,
            requires_grad=True,
            stores_grad=True
        )
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        # initialize A the same way as the default for nn.Linear and B to zero
        self.lora_A.gaussian(0.0, std)
        self.lora_B.set_value(0.0)
        self.scaling = tensor.Tensor(shape=(1,), requires_grad=False, stores_grad=False)
        self.scaling.set_value(1.0 * self.lora_alpha / self.r)

    def freeze_pretrained_weight(self, freeze: bool = True):
        # freeze weights of base layer
        self.base_layer.W.requires_grad = not freeze
        self.base_layer.W.stores_grad = not freeze
        if self.base_layer.b is not None:
            self.base_layer.b.requires_grad = not freeze
            self.base_layer.b.stores_grad = not freeze

    def forward(self, x):
        # forward
        if not self.merged:
            y1 = self.base_layer(x)
            y2 = autograd.dropout(x, self.lora_dropout)
            y2 = autograd.matmul(y2, autograd.transpose(self.lora_A, (1, 0)))
            y2 = autograd.matmul(y2, autograd.transpose(self.lora_B, (1, 0)))
            y2 = autograd.mul(y2, self.scaling)
            y = autograd.add(y1, y2)
            return y
        else:
            y = self.base_layer(x)
            return y

    def merge_weights(self, mode: bool = True):
        # Merge the weights
        if mode:
            if not self.merged:
                # Merge the weights and mark it
                delta = tensor.mult(self.lora_A.transpose((1, 0)), self.lora_B.transpose((1, 0))) * self.scaling
                self.base_layer.W.data += delta.data
                self.merged = True
        else:
            if self.merged:
                # Make sure that the weights are not merged
                delta = tensor.mult(self.lora_A.transpose((1, 0)), self.lora_B.transpose((1, 0))) * self.scaling
                self.base_layer.W.data -= delta.data
                self.merged = False

    def get_params(self):
        params = self.base_layer.get_params()
        params[self.lora_A.name] = self.lora_A
        params[self.lora_B.name] = self.lora_B
        return params

    def set_params(self, parameters):
        self.base_layer.set_params(parameters)
        self.lora_A.copy_from(parameters[self.lora_A.name])
        self.lora_B.copy_from(parameters[self.lora_B.name])