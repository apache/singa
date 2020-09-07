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
from singa import layer
from singa import model


class IMDBModel(model.Model):

    def __init__(self,
                 hidden_size,
                 mode='lstm',
                 return_sequences=False,
                 bidirectional="False",
                 num_layers=1):
        super().__init__()
        batch_first = True
        self.lstm = layer.CudnnRNN(hidden_size=hidden_size,
                                   batch_first=batch_first,
                                   rnn_mode=mode,
                                   return_sequences=return_sequences,
                                   num_layers=1,
                                   dropout=0.9,
                                   bidirectional=bidirectional)
        self.l1 = layer.Linear(64)
        self.l2 = layer.Linear(2)

    def forward(self, x):
        y = self.lstm(x)
        y = autograd.reshape(y, (y.shape[0], -1))
        y = self.l1(y)
        y = autograd.relu(y)
        y = self.l2(y)
        return y

    def train_one_batch(self, x, y):
        out = self.forward(x)
        loss = autograd.softmax_cross_entropy(out, y)
        self.optimizer(loss)
        return out, loss

    def set_opt(self, optimizer):
        self.optimizer = optimizer
