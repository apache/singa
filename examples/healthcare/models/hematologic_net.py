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
from singa import tensor
from singa import opt
from singa import device

import numpy as np


np_dtype = {"float16": np.float16, "float32": np.float32}

singa_dtype = {"float16": tensor.float16, "float32": tensor.float32}


class CNNModel(model.Model):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.input_size = 28
        self.dimension = 4
        self.num_classes = num_classes

        self.layer1 = layer.Conv2d(16, kernel_size=3, activation="RELU")
        self.bn1 = layer.BatchNorm2d()
        self.layer2 = layer.Conv2d(16, kernel_size=3, activation="RELU")
        self.bn2 = layer.BatchNorm2d()
        self.pooling2 = layer.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = layer.Conv2d(64, kernel_size=3, activation="RELU")
        self.bn3 = layer.BatchNorm2d()
        self.layer4 = layer.Conv2d(64, kernel_size=3, activation="RELU")
        self.bn4 = layer.BatchNorm2d()
        self.layer5 = layer.Conv2d(64, kernel_size=3, padding=1, activation="RELU")
        self.bn5 = layer.BatchNorm2d()
        self.pooling5 = layer.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = layer.Flatten()

        self.linear1 = layer.Linear(128)
        self.linear2 = layer.Linear(128)
        self.linear3 = layer.Linear(self.num_classes)

        self.relu = layer.ReLU()

        self.softmax_cross_entropy = layer.SoftMaxCrossEntropy()
        self.dropout = layer.Dropout(ratio=0.3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.pooling2(x)

        x = self.layer3(x)
        x = self.bn3(x)
        x = self.layer4(x)
        x = self.bn4(x)
        x = self.layer5(x)
        x = self.bn5(x)
        x = self.pooling5(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

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


def create_model(**kwargs):
    """Constructs a CNN model.

    Args:
        pretrained (bool): If True, returns a pre-trained model.

    Returns:
        The created CNN model.
    """
    model = CNNModel(**kwargs)

    return model


__all__ = ['CNNModel', 'create_model']