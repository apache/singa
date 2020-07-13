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

from singa import autograd, layer, model, tensor, device, opt

class QAModel(model.Model):

    def __init__(self,
                 hidden_size,
                 num_layers=1,
                 rnn_mode="lstm",
                 batch_first=True,
                 return_sequences=True):
        super(QAModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm_q = layer.CudnnRNN(hidden_size=hidden_size,return_sequences=return_sequences)
        self.lstm_a = layer.CudnnRNN(hidden_size=hidden_size,return_sequences=return_sequences)
        self.optimizer = opt.SGD()

    def forward(self, q, a_batch):
        q = self.lstm_q(q)  # bs, seq, Hidden*2
        a_batch = self.lstm_a(a_batch)  # 2bs, seq, hidden*2
        bs_a = q.shape[0]
        a_pos, a_neg = autograd.split(a_batch, 0, [bs_a, bs_a])
        q = autograd.reshape(q, (-1,self.hidden_size))
        a_pos = autograd.reshape(a_pos, (-1,self.hidden_size))
        a_neg = autograd.reshape(a_neg, (-1,self.hidden_size))
        # a_pos = autograd.mul(q,a_pos)
        # a_neg = autograd.mul(q,a_neg)
        # return a_pos, a_neg
        sim_pos = autograd.cossim(q, a_pos)
        sim_neg = autograd.cossim(q, a_neg)
        return sim_pos, sim_neg

    def train_one_batch(self, q, a):
        out = self.forward(q, a)
        loss = autograd.qa_lstm_loss(out[0], out[1])
        self.optimizer(loss)
        return out, loss


class MLP(model.Model):

    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = layer.Linear(500)
        self.relu = layer.ReLU()
        self.linear2 = layer.Linear(2)
        self.optimizer=opt.SGD()

    def forward(self, q, a):
        q = autograd.reshape(q, (q.shape[0], -1))
        a = autograd.reshape(a, (q.shape[0], -1))
        qa = autograd.cat([q, a], 1)
        y = self.linear1(qa)
        y = self.relu(y)
        y = self.linear2(y)
        return y

    def train_one_batch(self, q, a, y):
        out = self.forward(q, a)
        loss = autograd.softmax_cross_entropy(out, y)
        self.optimizer(loss)
        return out, loss


if __name__ == "__main__":
    m = QAModel(2,return_sequences=False)
    dev = device.create_cuda_gpu_on(7)

    tq = tensor.random((2, 3, 4),dev)
    ta = tensor.random((2 * 2, 3, 4),dev)

    m.train()
    for i in range(10):
        out = m.forward(tq, ta)
        loss = autograd.qa_lstm_loss(out[0], out[1])
        m.optimizer(loss)
        print(loss)