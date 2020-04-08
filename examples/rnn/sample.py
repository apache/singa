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
'''Sample characters from the pre-trained model'''

from __future__ import division
from __future__ import print_function
from builtins import range
import sys
import numpy as np
import argparse
try:
    import pickle
except ImportError:
    import cPickle as pickle

from singa import layer
from singa import tensor
from singa import device


def sample(model_path, nsamples=100, seed_text='', do_sample=True):
    with open(model_path, 'rb') as fd:
        d = pickle.load(fd)
        rnn_w = tensor.from_numpy(d['rnn_w'])
        idx_to_char = d['idx_to_char']
        char_to_idx = d['char_to_idx']
        vocab_size = len(idx_to_char)
        dense_w = tensor.from_numpy(d['dense_w'])
        dense_b = tensor.from_numpy(d['dense_b'])
        hidden_size = d['hidden_size']
        num_stacks = d['num_stacks']
        dropout = d['dropout']

    cuda = device.create_cuda_gpu()
    rnn = layer.LSTM(name='lstm', hidden_size=hidden_size,
                     num_stacks=num_stacks, dropout=dropout,
                     input_sample_shape=(len(idx_to_char),))
    rnn.to_device(cuda)
    rnn.param_values()[0].copy_data(rnn_w)
    dense = layer.Dense('dense', vocab_size, input_sample_shape=(hidden_size,))
    dense.to_device(cuda)
    dense.param_values()[0].copy_data(dense_w)
    dense.param_values()[1].copy_data(dense_b)
    hx = tensor.Tensor((num_stacks, 1, hidden_size), cuda)
    cx = tensor.Tensor((num_stacks, 1, hidden_size), cuda)
    hx.set_value(0.0)
    cx.set_value(0.0)
    if len(seed_text) > 0:
        for c in seed_text:
            x = np.zeros((1, vocab_size), dtype=np.float32)
            x[0, char_to_idx[c]] = 1
            tx = tensor.from_numpy(x)
            tx.to_device(cuda)
            inputs = [tx, hx, cx]
            outputs = rnn.forward(False, inputs)
            y = dense.forward(False, outputs[0])
            y = tensor.softmax(y)
            hx = outputs[1]
            cx = outputs[2]
        sys.stdout.write(seed_text)
    else:
        y = tensor.Tensor((1, vocab_size), cuda)
        y.set_value(1.0 / vocab_size)

    for i in range(nsamples):
        y.to_host()
        prob = tensor.to_numpy(y)[0]
        if do_sample:
            cur = np.random.choice(vocab_size, 1, p=prob)[0]
        else:
            cur = np.argmax(prob)
        sys.stdout.write(idx_to_char[cur])
        x = np.zeros((1, vocab_size), dtype=np.float32)
        x[0, cur] = 1
        tx = tensor.from_numpy(x)
        tx.to_device(cuda)
        inputs = [tx, hx, cx]
        outputs = rnn.forward(False, inputs)
        y = dense.forward(False, outputs[0])
        y = tensor.softmax(y)
        hx = outputs[1]
        cx = outputs[2]
    print('')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='sample chars from char-rnn')
    parser.add_argument('model', help='the model checkpoint file')
    parser.add_argument('n', type=int, help='num of characters to sample')
    parser.add_argument('--seed', help='seed text string which warms up the '
                        ' rnn states for sampling', default='')
    args = parser.parse_args()
    assert args.n > 0, 'n must > 0'
    sample(args.model, args.n, seed_text=args.seed)
