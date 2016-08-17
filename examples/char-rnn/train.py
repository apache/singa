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
import cPickle as pickle
import numpy as np
import argparse

# sys.path.append(os.path.join(os.path.dirname(__file__), '../../build/python'))
from singa import layer
from singa import loss
from singa import device
from singa import tensor
from singa import optimizer
from singa import initializer
from singa.proto import model_pb2
from singa import utils


class Data(object):

    def __init__(self, fpath, batch_size=32, seq_length=100, train_ratio=0.8):
        '''Data object for loading a plain text file.

        Args:
            fpath, path to the text file.
            train_ratio, split the text file into train and test sets, where
                train_ratio of the characters are in the train set.
        '''
        self.raw_data = open(fpath, 'r').read()  # read text file
        chars = list(set(self.raw_data))
        self.vocab_size = len(chars)
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        data = [self.char_to_idx[c] for c in self.raw_data]
        # seq_length + 1 for the data + label
        nsamples = len(data) / (1 + seq_length)
        data = data[0:nsamples * (1 + seq_length)]
        data = np.asarray(data, dtype=np.int32)
        data = np.reshape(data, (-1, seq_length + 1))
        # shuffle all sequences
        np.random.shuffle(data)
        self.train_dat = data[0:int(data.shape[0]*train_ratio)]
        self.num_train_batch = self.train_dat.shape[0] / batch_size
        self.val_dat = data[self.train_dat.shape[0]:]
        self.num_test_batch = self.val_dat.shape[0] / batch_size
        print 'train dat', self.train_dat.shape
        print 'val dat', self.val_dat.shape


def numpy2tensors(npx, npy, dev):
    '''batch, seq, dim -- > seq, batch, dim'''
    tmpx = np.swapaxes(npx, 0, 1)
    tmpy = np.swapaxes(npy, 0, 1)
    inputs = []
    labels = []
    for t in range(tmpx.shape[0]):
        x = tensor.from_numpy(tmpx[t])
        y = tensor.from_numpy(tmpy[t])
        x.to_device(dev)
        y.to_device(dev)
        inputs.append(x)
        labels.append(y)
    return inputs, labels


def convert(batch, batch_size, seq_length, vocab_size, dev):
    '''convert a batch of data into a sequence of input tensors'''
    y = batch[:, 1:]
    x1 = batch[:, :seq_length]
    x = np.zeros((batch_size, seq_length, vocab_size), dtype=np.float32)
    for b in range(batch_size):
        for t in range(seq_length):
            c = x1[b, t]
            x[b, t, c] = 1
    return numpy2tensors(x, y, dev)


def get_lr(epoch):
    return 0.001 / float(1 << (epoch / 50))


def train(data, max_epoch, hidden_size=100, seq_length=100, batch_size=16,
          num_stacks=1, dropout=0.5, model_path='model'):
    # SGD with L2 gradient normalization
    opt = optimizer.RMSProp(constraint=optimizer.L2Constraint(5))
    cuda = device.create_cuda_gpu()
    rnn = layer.LSTM(
        name='lstm',
        hidden_size=hidden_size,
        num_stacks=num_stacks,
        dropout=dropout,
        input_sample_shape=(
            data.vocab_size,
        ))
    rnn.to_device(cuda)
    print 'created rnn'
    rnn_w = rnn.param_values()[0]
    rnn_w.uniform(-0.08, 0.08)  # init all rnn parameters
    print 'rnn weight l1 = %f' % (rnn_w.l1())
    dense = layer.Dense(
        'dense',
        data.vocab_size,
        input_sample_shape=(
            hidden_size,
        ))
    dense.to_device(cuda)
    dense_w = dense.param_values()[0]
    dense_b = dense.param_values()[1]
    print 'dense w ', dense_w.shape
    print 'dense b ', dense_b.shape
    initializer.uniform(dense_w, dense_w.shape[0], 0)
    print 'dense weight l1 = %f' % (dense_w.l1())
    dense_b.set_value(0)
    print 'dense b l1 = %f' % (dense_b.l1())

    g_dense_w = tensor.Tensor(dense_w.shape, cuda)
    g_dense_b = tensor.Tensor(dense_b.shape, cuda)

    lossfun = loss.SoftmaxCrossEntropy()
    for epoch in range(max_epoch):
        train_loss = 0
        for b in range(data.num_train_batch):
            batch = data.train_dat[b * batch_size: (b + 1) * batch_size]
            inputs, labels = convert(batch, batch_size, seq_length,
                                     data.vocab_size, cuda)
            inputs.append(tensor.Tensor())
            inputs.append(tensor.Tensor())

            outputs = rnn.forward(model_pb2.kTrain, inputs)[0:-2]
            grads = []
            batch_loss = 0
            g_dense_w.set_value(0.0)
            g_dense_b.set_value(0.0)
            for output, label in zip(outputs, labels):
                act = dense.forward(model_pb2.kTrain, output)
                lvalue = lossfun.forward(model_pb2.kTrain, act, label)
                batch_loss += lvalue.l1()
                grad = lossfun.backward()
                grad /= batch_size
                grad, gwb = dense.backward(model_pb2.kTrain, grad)
                grads.append(grad)
                g_dense_w += gwb[0]
                g_dense_b += gwb[1]
                # print output.l1(), act.l1()
            utils.update_progress(
                b * 1.0 / data.num_train_batch, 'training loss = %f' %
                (batch_loss / seq_length))
            train_loss += batch_loss

            grads.append(tensor.Tensor())
            grads.append(tensor.Tensor())
            g_rnn_w = rnn.backward(model_pb2.kTrain, grads)[1][0]
            dense_w, dense_b = dense.param_values()
            opt.apply_with_lr(epoch, get_lr(epoch), g_rnn_w, rnn_w, 'rnnw')
            opt.apply_with_lr(
                epoch, get_lr(epoch),
                g_dense_w, dense_w, 'dense_w')
            opt.apply_with_lr(
                epoch, get_lr(epoch),
                g_dense_b, dense_b, 'dense_b')
        print '\nEpoch %d, train loss is %f' % \
            (epoch, train_loss / data.num_train_batch / seq_length)

        eval_loss = 0
        for b in range(data.num_test_batch):
            batch = data.val_dat[b * batch_size: (b + 1) * batch_size]
            inputs, labels = convert(batch, batch_size, seq_length,
                                     data.vocab_size, cuda)
            inputs.append(tensor.Tensor())
            inputs.append(tensor.Tensor())
            outputs = rnn.forward(model_pb2.kEval, inputs)[0:-2]
            for output, label in zip(outputs, labels):
                output = dense.forward(model_pb2.kEval, output)
                eval_loss += lossfun.forward(model_pb2.kEval,
                                             output, label).l1()
        print 'Epoch %d, evaluation loss is %f' % \
            (epoch, eval_loss / data.num_test_batch / seq_length)

        if (epoch + 1) % 30 == 0:
            # checkpoint the file model
            with open('%s_%d.bin' % (model_path, epoch), 'wb') as fd:
                print 'saving model to %s' % model_path
                d = {}
                for name, w in zip(
                        ['rnn_w', 'dense_w', 'dense_b'],
                        [rnn_w, dense_w, dense_b]):
                    w.to_host()
                    d[name] = tensor.to_numpy(w)
                    w.to_device(cuda)
                d['idx_to_char'] = data.idx_to_char
                d['char_to_idx'] = data.char_to_idx
                d['hidden_size'] = hidden_size
                d['num_stacks'] = num_stacks
                d['dropout'] = dropout

                pickle.dump(d, fd)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train multi-stack LSTM for '
        'modeling  character sequence from plain text files')
    parser.add_argument('data', type=str, help='training file')
    parser.add_argument('-b', type=int, default=32, help='batch_size')
    parser.add_argument('-l', type=int, default=64, help='sequence length')
    parser.add_argument('-d', type=int, default=128, help='hidden size')
    parser.add_argument('-s', type=int, default=2, help='num of stacks')
    parser.add_argument('-m', type=int, default=50, help='max num of epoch')
    args = parser.parse_args()
    data = Data(args.data, batch_size=args.b, seq_length=args.l)
    train(data, args.m,  hidden_size=args.d, num_stacks=args.s,
          seq_length=args.l, batch_size=args.b)
