#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from singa import layer
from singa import model
import singa.tensor as tensor
from singa import autograd
from singa.tensor import Tensor


class CPLayer(layer.Layer):
    def __init__(self, prototype_count=2, temp=10.0):
        super(CPLayer, self).__init__()
        self.prototype_count = prototype_count
        self.temp = temp

    def initialize(self, x):
        self.feature_dim = x.shape[1]
        self.prototype = tensor.random(
            (self.feature_dim, self.prototype_count), device=x.device
        )

    def forward(self, feat):
        self.device_check(feat, self.prototype)
        self.dtype_check(feat, self.prototype)

        feat_sq = autograd.mul(feat, feat)
        feat_sq_sum = autograd.reduce_sum(feat_sq, axes=[1], keepdims=1)
        feat_sq_sum_tile = autograd.tile(feat_sq_sum, repeats=[1, self.feature_dim])

        prototype_sq = autograd.mul(self.prototype, self.prototype)
        prototype_sq_sum = autograd.reduce_sum(prototype_sq, axes=[0], keepdims=1)
        prototype_sq_sum_tile = autograd.tile(prototype_sq_sum, repeats=feat.shape[0])

        cross_term = autograd.matmul(feat, self.prototype)
        cross_term_scale = Tensor(
            shape=cross_term.shape, device=cross_term.device, requires_grad=False
        ).set_value(-2)
        cross_term_scaled = autograd.mul(cross_term, cross_term_scale)

        dist = autograd.add(feat_sq_sum_tile, prototype_sq_sum_tile)
        dist = autograd.add(dist, cross_term_scaled)

        logits_coeff = (
            tensor.ones((feat.shape[0], self.prototype.shape[1]), device=feat.device)
            * -1.0
            / self.temp
        )
        logits_coeff.requires_grad = False
        logits = autograd.mul(logits_coeff, dist)

        return logits

    def get_params(self):
        return {self.prototype.name: self.prototype}

    def set_params(self, parameters):
        self.prototype.copy_from(parameters[self.prototype.name])


class CPL(model.Model):

    def __init__(
        self,
        backbone: model.Model,
        prototype_count=2,
        lamb=0.5,
        temp=10,
        label=None,
        prototype_weight=None,
    ):
        super(CPL, self).__init__()
        # config
        self.lamb = lamb
        self.prototype_weight = prototype_weight
        self.prototype_label = label

        # layer
        self.backbone = backbone
        self.cplayer = CPLayer(prototype_count=prototype_count, temp=temp)
        # optimizer
        self.softmax_cross_entropy = layer.SoftMaxCrossEntropy()

    def forward(self, x):
        feat = self.backbone.forward(x)
        logits = self.cplayer(feat)
        return logits

    def train_one_batch(self, x, y, dist_option, spars):
        out = self.forward(x)
        loss = self.softmax_cross_entropy(out, y)
        self.optimizer(loss)
        return out, loss

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer


class CNN(model.Model):

    def __init__(self, num_classes=10, num_channels=1):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.input_size = 28
        self.dimension = 4
        self.conv1 = layer.Conv2d(num_channels, 20, 5, padding=0, activation="RELU")
        self.conv2 = layer.Conv2d(20, 50, 5, padding=0, activation="RELU")
        self.linear1 = layer.Linear(500)
        self.linear2 = layer.Linear(num_classes)
        self.pooling1 = layer.MaxPool2d(2, 2, padding=0)
        self.pooling2 = layer.MaxPool2d(2, 2, padding=0)
        self.relu = layer.ReLU()
        self.flatten = layer.Flatten()
        self.softmax_cross_entropy = layer.SoftMaxCrossEntropy()

    def forward(self, x):
        y = self.conv1(x)
        y = self.pooling1(y)
        y = self.conv2(y)
        y = self.pooling2(y)
        y = self.flatten(y)
        y = self.linear1(y)
        y = self.relu(y)
        y = self.linear2(y)
        return y

    def train_one_batch(self, x, y, dist_option, spars):
        out = self.forward(x)
        loss = self.softmax_cross_entropy(out, y)

        if dist_option == 'plain':
            self.optimizer(loss)
        elif dist_option == 'half':
            self.optimizer.backward_and_update_half(loss)
        elif dist_option == 'partialUpdate':
            self.optimizer.backward_and_partial_update(loss)
        elif dist_option == 'sparseTopK':
            self.optimizer.backward_and_sparse_update(loss,
                                                      topK=True,
                                                      spars=spars)
        elif dist_option == 'sparseThreshold':
            self.optimizer.backward_and_sparse_update(loss,
                                                      topK=False,
                                                      spars=spars)
        return out, loss

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

def create_cnn_model(pretrained=False, **kwargs):
    """Constructs a CNN model.

    Args:
        pretrained (bool): If True, returns a pre-trained model.

    Returns:
        The created CNN model.
    """
    model = CNN(**kwargs)

    return model

def create_model(backbone, prototype_count=2, lamb=0.5, temp=10.0):
    model = CPL(backbone, prototype_count=prototype_count, lamb=lamb, temp=temp)
    return model


__all__ = ["CPL", "CNN", "create_cnn_model", "create_model"]
