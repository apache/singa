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

from singa import autograd
from singa import module


class AlexNet(module.Module):

    def __init__(self, num_classes=10, num_channels=1):
        super(AlexNet, self).__init__()
        self.num_classes = num_classes
        self.input_size = 224
        self.dimension = 4
        self.conv1 = autograd.Conv2d(num_channels, 64, 11, stride=4, padding=2)
        self.conv2 = autograd.Conv2d(64, 192, 5, padding=2)
        self.conv3 = autograd.Conv2d(192, 384, 3, padding=1)
        self.conv4 = autograd.Conv2d(384, 256, 3, padding=1)
        self.conv5 = autograd.Conv2d(256, 256, 3, padding=1)
        self.linear1 = autograd.Linear(1024, 4096)
        self.linear2 = autograd.Linear(4096, 4096)
        self.linear3 = autograd.Linear(4096, num_classes)
        self.pooling1 = autograd.MaxPool2d(2, 2, padding=0)
        self.pooling2 = autograd.MaxPool2d(2, 2, padding=0)
        self.pooling3 = autograd.MaxPool2d(2, 2, padding=0)
        self.avg_pooling1 = autograd.AvgPool2d(3, 2, padding=0)

    def forward(self, x):
        y = self.conv1(x)
        y = autograd.relu(y)
        y = self.pooling1(y)
        y = self.conv2(y)
        y = autograd.relu(y)
        y = self.pooling2(y)
        y = self.conv3(y)
        y = autograd.relu(y)
        y = self.conv4(y)
        y = autograd.relu(y)
        y = self.conv5(y)
        y = autograd.relu(y)
        y = self.pooling3(y)
        y = self.avg_pooling1(y)
        y = autograd.flatten(y)
        y = autograd.dropout(y)
        y = self.linear1(y)
        y = autograd.relu(y)
        y = autograd.dropout(y)
        y = self.linear2(y)
        y = autograd.relu(y)
        y = self.linear3(y)
        return y

    def loss(self, out, ty):
        return autograd.softmax_cross_entropy(out, ty)

    def optim(self, loss, dist_option, spars):
        if dist_option == 'fp32':
            self.optimizer.backward_and_update(loss)
        elif dist_option == 'fp16':
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

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer


def create_model(pretrained=False, **kwargs):
    """Constructs a AlexNet model.

    Args:
        pretrained (bool): If True, returns a model pre-trained
    """
    model = AlexNet(**kwargs)

    return model


__all__ = ['AlexNet', 'create_model']
