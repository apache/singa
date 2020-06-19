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
'''Train a Char-RNN model using plain text files.
The model is created following https://github.com/karpathy/char-rnn
The train file could be any text file,
e.g., http://cs.stanford.edu/people/karpathy/char-rnn/
'''

from __future__ import division
from __future__ import print_function
from builtins import range
import numpy as np
import sys
import argparse
from tqdm import tqdm

from singa import device
from singa import tensor
from singa import autograd
from singa import layer
from singa import model
from singa import opt


class CharRNN(model.Model):

    def __init__(self, vocab_size, hidden_size=32):
        super(CharRNN, self).__init__()
        self.rnn = layer.LSTM(vocab_size, hidden_size)
        self.cat = layer.Cat()
        self.reshape1 = layer.Reshape()
        self.dense = layer.Linear(hidden_size, vocab_size)
        self.reshape2 = layer.Reshape()
        self.softmax_cross_entropy = layer.SoftMaxCrossEntropy()
        self.optimizer = opt.SGD(0.01)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

    def reset_states(self, dev):
        self.hx.to_device(dev)
        self.cx.to_device(dev)
        self.hx.set_value(0.0)
        self.cx.set_value(0.0)

    def initialize(self, inputs):
        batchsize = inputs[0].shape[0]
        self.hx = tensor.Tensor((batchsize, self.hidden_size))
        self.cx = tensor.Tensor((batchsize, self.hidden_size))
        self.reset_states(inputs[0].device)

    def forward(self, inputs):
        x, hx, cx = self.rnn(inputs, (self.hx, self.cx))
        self.hx.copy_data(hx)
        self.cx.copy_data(cx)
        x = self.cat(x)
        x = self.reshape1(x, (-1, self.hidden_size))
        return self.dense(x)

    def train_one_batch(self, x, y):
        out = self.forward(x)
        y = self.reshape2(y, (-1, 1))
        loss = self.softmax_cross_entropy(out, y)
        self.optimizer(loss)
        return out, loss

    def get_states(self):
        ret = super().get_states()
        ret[self.hx.name] = self.hx
        ret[self.cx.name] = self.cx
        return ret

    def set_states(self, states):
        self.hx.copy_from(states[self.hx.name])
        self.hx.copy_from(states[self.hx.name])
        super().set_states(states)


class Data(object):

    def __init__(self, fpath, batch_size=32, seq_length=100, train_ratio=0.8):
        '''Data object for loading a plain text file.

        Args:
            fpath, path to the text file.
            train_ratio, split the text file into train and test sets, where
                train_ratio of the characters are in the train set.
        '''
        self.raw_data = open(fpath, 'r',
                             encoding='iso-8859-1').read()  # read text file
        chars = list(set(self.raw_data))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        data = [self.char_to_idx[c] for c in self.raw_data]
        # seq_length + 1 for the data + label
        nsamples = len(data) // (1 + seq_length)
        data = data[0:nsamples * (1 + seq_length)]
        data = np.asarray(data, dtype=np.int32)
        data = np.reshape(data, (-1, seq_length + 1))
        # shuffle all sequences
        np.random.shuffle(data)
        self.train_dat = data[0:int(data.shape[0] * train_ratio)]
        self.num_train_batch = self.train_dat.shape[0] // batch_size
        self.val_dat = data[self.train_dat.shape[0]:]
        self.num_test_batch = self.val_dat.shape[0] // batch_size
        print('train dat', self.train_dat.shape)
        print('val dat', self.val_dat.shape)


def numpy2tensors(npx, npy, dev, inputs=None, labels=None):
    '''batch, seq, dim -- > seq, batch, dim'''
    tmpy = np.swapaxes(npy, 0, 1).reshape((-1, 1))
    if labels:
        labels.copy_from_numpy(tmpy)
    else:
        labels = tensor.from_numpy(tmpy)
    labels.to_device(dev)
    tmpx = np.swapaxes(npx, 0, 1)
    inputs_ = []
    for t in range(tmpx.shape[0]):
        if inputs:
            inputs[t].copy_from_numpy(tmpx[t])
        else:
            x = tensor.from_numpy(tmpx[t])
            x.to_device(dev)
            inputs_.append(x)
    if not inputs:
        inputs = inputs_
    return inputs, labels


def convert(batch,
            batch_size,
            seq_length,
            vocab_size,
            dev,
            inputs=None,
            labels=None):
    '''convert a batch of data into a sequence of input tensors'''
    y = batch[:, 1:]
    x1 = batch[:, :seq_length]
    x = np.zeros((batch_size, seq_length, vocab_size), dtype=np.float32)
    for b in range(batch_size):
        for t in range(seq_length):
            c = x1[b, t]
            x[b, t, c] = 1
    return numpy2tensors(x, y, dev, inputs, labels)


def sample(model, data, dev, nsamples=100, use_max=False):
    while True:
        cmd = input('Do you want to sample text from the model [y/n]')
        if cmd == 'n':
            return
        else:
            seed = input('Please input some seeding text, e.g., #include <c: ')
            inputs = []
            for c in seed:
                x = np.zeros((1, data.vocab_size), dtype=np.float32)
                x[0, data.char_to_idx[c]] = 1
                tx = tensor.from_numpy(x)
                tx.to_device(dev)
                inputs.append(tx)
            model.reset_states(dev)
            outputs = model(inputs)
            y = tensor.softmax(outputs[-1])
            sys.stdout.write(seed)
            for i in range(nsamples):
                prob = tensor.to_numpy(y)[0]
                if use_max:
                    cur = np.argmax(prob)
                else:
                    cur = np.random.choice(data.vocab_size, 1, p=prob)[0]
                sys.stdout.write(data.idx_to_char[cur])
                x = np.zeros((1, data.vocab_size), dtype=np.float32)
                x[0, cur] = 1
                tx = tensor.from_numpy(x)
                tx.to_device(dev)
                outputs = model([tx])
                y = tensor.softmax(outputs[-1])


def evaluate(model, data, batch_size, seq_length, dev, inputs, labels):
    model.eval()
    val_loss = 0.0
    for b in range(data.num_test_batch):
        batch = data.val_dat[b * batch_size:(b + 1) * batch_size]
        inputs, labels = convert(batch, batch_size, seq_length, data.vocab_size,
                                 dev, inputs, labels)
        model.reset_states(dev)
        y = model(inputs)
        loss = autograd.softmax_cross_entropy(y, labels)[0]
        val_loss += tensor.to_numpy(loss)[0]
    print('            validation loss is %f' %
          (val_loss / data.num_test_batch / seq_length))


def train(data,
          max_epoch,
          hidden_size=100,
          seq_length=100,
          batch_size=16,
          model_path='model'):
    # SGD with L2 gradient normalization
    cuda = device.create_cuda_gpu()
    model = CharRNN(data.vocab_size, hidden_size)
    model.graph(True, False)

    inputs, labels = None, None

    for epoch in range(max_epoch):
        model.train()
        train_loss = 0
        for b in tqdm(range(data.num_train_batch)):
            batch = data.train_dat[b * batch_size:(b + 1) * batch_size]
            inputs, labels = convert(batch, batch_size, seq_length,
                                     data.vocab_size, cuda, inputs, labels)
            out, loss = model(inputs, labels)
            model.reset_states(cuda)
            train_loss += tensor.to_numpy(loss)[0]

        print('\nEpoch %d, train loss is %f' %
              (epoch, train_loss / data.num_train_batch / seq_length))

        evaluate(model, data, batch_size, seq_length, cuda, inputs, labels)
        sample(model, data, cuda)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train multi-stack LSTM for '
        'modeling  character sequence from plain text files')
    parser.add_argument('data', type=str, help='training file')
    parser.add_argument('-b', type=int, default=32, help='batch_size')
    parser.add_argument('-l', type=int, default=64, help='sequence length')
    parser.add_argument('-d', type=int, default=128, help='hidden size')
    parser.add_argument('-m', type=int, default=50, help='max num of epoch')
    args = parser.parse_args()
    data = Data(args.data, batch_size=args.b, seq_length=args.l)
    train(data,
          args.m,
          hidden_size=args.d,
          seq_length=args.l,
          batch_size=args.b)
