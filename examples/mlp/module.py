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

from singa import module
from singa import autograd
from singa import tensor
from singa.tensor import Tensor

class MLP(module.Module):

    def __init__(self, data_size=10, perceptron_size=100, num_classes=10):
        super(MLP, self).__init__()
        self.num_classes = num_classes
        self.dimension = 2

        self.w0 = Tensor(shape=(data_size, perceptron_size),
                         requires_grad=True,
                         stores_grad=True)
        self.w0.gaussian(0.0, 0.1)
        self.b0 = Tensor(shape=(perceptron_size,),
                         requires_grad=True,
                         stores_grad=True)
        self.b0.set_value(0.0)

        self.w1 = Tensor(shape=(perceptron_size, num_classes),
                         requires_grad=True,
                         stores_grad=True)
        self.w1.gaussian(0.0, 0.1)
        self.b1 = Tensor(shape=(num_classes,),
                         requires_grad=True,
                         stores_grad=True)
        self.b1.set_value(0.0)

    def forward(self, inputs):
        x = autograd.matmul(inputs, self.w0)
        x = autograd.add_bias(x, self.b0)
        x = autograd.relu(x)
        x = autograd.matmul(x, self.w1)
        x = autograd.add_bias(x, self.b1)
        return x

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
    """Constructs a CNN model.

    Args:
        pretrained (bool): If True, returns a model pre-trained
    """
    model = MLP(**kwargs)

    return model


__all__ = ['MLP', 'create_model']
