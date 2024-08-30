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

from singa import layer
from singa import model
from singa import autograd


class GAN_MLP(model.Model):

    def __init__(self, noise_size=100, feature_size=784, hidden_size=128):
        super(GAN_MLP, self).__init__()
        self.noise_size = noise_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size

        # Generative Net
        self.gen_net_fc_0 = layer.Linear(self.hidden_size)
        self.gen_net_relu_0 = layer.ReLU()
        self.gen_net_fc_1 = layer.Linear(self.feature_size)
        self.gen_net_sigmoid_1 = layer.Sigmoid()

        # Discriminative Net
        self.dis_net_fc_0 = layer.Linear(self.hidden_size)
        self.dis_net_relu_0 = layer.ReLU()
        self.dis_net_fc_1 = layer.Linear(1)
        self.dis_net_sigmoid_1 = layer.Sigmoid()
        self.binary_cross_entropy = layer.BinaryCrossEntropy()

    def forward(self, x):
        # Cascaded Net
        y = self.forward_gen(x)
        y = self.forward_dis(y)
        return y

    def forward_dis(self, x):
        # Discriminative Net
        y = self.dis_net_fc_0(x)
        y = self.dis_net_relu_0(y)
        y = self.dis_net_fc_1(y)
        y = self.dis_net_sigmoid_1(y)
        return y

    def forward_gen(self, x):
        # Generative Net
        y = self.gen_net_fc_0(x)
        y = self.gen_net_relu_0(y)
        y = self.gen_net_fc_1(y)
        y = self.gen_net_sigmoid_1(y)
        return y

    def train_one_batch(self, x, y):
        # Training the Generative Net
        out = self.forward(x)
        loss = self.binary_cross_entropy(out, y)
        # Only update the Generative Net
        for p, g in autograd.backward(loss):
            if "gen_net" in p.name:
                self.optimizer.apply(p.name, p, g)
        return out, loss

    def train_one_batch_dis(self, x, y):
        # Training the Discriminative Net
        out = self.forward_dis(x)
        loss = self.binary_cross_entropy(out, y)
        # Only update the Discriminative Net
        for p, g in autograd.backward(loss):
            if "dis_net" in p.name:
                self.optimizer.apply(p.name, p, g)
        self.optimizer(loss)
        return out, loss

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer


def create_model(pretrained=False, **kwargs):
    """Constructs a CNN model.

    Args:
        pretrained (bool): If True, returns a model pre-trained
    """
    model = GAN_MLP(**kwargs)

    return model


__all__ = ['GAN_MLP', 'create_model']
