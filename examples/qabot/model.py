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
from singa import layer
from singa import model
from singa import tensor
from singa import device

class QAModel(model.Model):
    def __init__(self, hidden_size, num_layers=1, rnn_mode="lstm", batch_first=True):
        super(QAModel, self).__init__()
        self.lstm_q = layer.CudnnRNN(hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   bidirectional=True,
                                   return_sequences=False,
                                   rnn_mode=rnn_mode,
                                   batch_first=batch_first)
        self.lstm_a = layer.CudnnRNN(hidden_size=hidden_size,
                                   num_layers=num_layers,
                                   bidirectional=True,
                                   return_sequences=False,
                                   rnn_mode=rnn_mode,
                                   batch_first=batch_first)

    def forward(self, q, a_batch):
        q = self.lstm_q(q) # BS, Hidden*2
        a_batch = self.lstm_a(a_batch) # {2, hidden*2}
        bs_a = int(a_batch.shape[0]/2) # cut concated a-a+ to half and half
        a_pos, a_neg = autograd.split(a_batch, 0, [bs_a,bs_a])
        sim_pos = autograd.cossim(q, a_pos)
        sim_neg = autograd.cossim(q, a_neg)
        return sim_pos, sim_neg

    def train_one_batch(self, q, a):
        out = self.forward(q, a)
        loss = autograd.qa_lstm_loss(out[0], out[1])
        self.optimizer.backward_and_update(loss)
        return out, loss



class MLP(model.Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = layer.Linear(500)
        self.relu = layer.ReLU()
        self.linear2 = layer.Linear(2)

    def forward(self, q, a):
        q=autograd.reshape(q, (q.shape[0], -1))
        a=autograd.reshape(a, (q.shape[0], -1))
        qa=autograd.cat([q,a], 1)
        y=self.linear1(qa)
        y=self.relu(y)
        y=self.linear2(y)
        return y

    def train_one_batch(self, q, a, y):
        out = self.forward(q, a)
        loss = autograd.softmax_cross_entropy(out, y)
        self.optimizer.backward_and_update(loss)
        return out, loss
