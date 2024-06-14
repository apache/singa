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

import pickle
import os
import sys
import numpy as np
from singa import tensor
from singa import device
from singa import opt
from imdb_data import pad_batch_2vec, preprocessed_imdb_data_fp
from imdb_model import IMDBModel
import argparse

if not os.path.isfile(preprocessed_imdb_data_fp):
    sys.exit(
        "Imdb dataset is not found, run python3 examples/rnn/imdb_data.py to prepare data"
    )

# load preprocessed data
imdb_processed = None
with open(preprocessed_imdb_data_fp, 'rb') as handle:
    imdb_processed = pickle.load(handle)

# use argparse to get command config: max_epoch, model, data, etc. for single gpu training
parser = argparse.ArgumentParser()
parser.add_argument('-m',
                    '--max-epoch',
                    default=5,
                    type=int,
                    help='maximum epochs',
                    dest='max_epoch')
parser.add_argument('-b',
                    '--batch-size',
                    default=128,
                    type=int,
                    help='batch size',
                    dest='bs')
parser.add_argument('-l',
                    '--learning-rate',
                    default=0.01,
                    type=float,
                    help='initial learning rate',
                    dest='lr')
# determine which gpu to use
parser.add_argument('-i',
                    '--device-id',
                    default=0,
                    type=int,
                    help='which GPU to use',
                    dest='device_id')
# training params
parser.add_argument('--mode',
                    default='lstm',
                    help='relu, tanh, lstm, gru',
                    dest='mode')
parser.add_argument('-s',
                    '--return-sequences',
                    default=False,
                    action='store_true',
                    help='return sequences',
                    dest='return_sequences')
parser.add_argument('-d',
                    '--bidirectional',
                    default=False,
                    action='store_true',
                    help='bidirectional lstm',
                    dest='bidirectional')
parser.add_argument('-n',
                    '--num-layers',
                    default=2,
                    type=int,
                    help='num layers',
                    dest='num_layers')

args = parser.parse_args()

# parameters
seq_limit = 50
embed_size = 300
hid = 32

# gpu device
dev = device.create_cuda_gpu_on(args.device_id)

# create placeholder
tx = tensor.Tensor((args.bs, seq_limit, embed_size), dev, tensor.float32)
ty = tensor.Tensor((args.bs, 2), dev, tensor.float32)
tx.gaussian(0, 1)
ty.gaussian(0, 1)

# create model
m = IMDBModel(hid,
              mode=args.mode,
              return_sequences=args.return_sequences,
              bidirectional=args.bidirectional,
              num_layers=args.num_layers)
m.set_opt(opt.SGD(args.lr, 0.9))

m.compile([tx], is_train=True, use_graph=True, sequential=False)

# training
m.train()
x_train, y_onehot_train, y_train = pad_batch_2vec(
    imdb_processed['train'], seq_limit, imdb_processed['embed_weights'])
x_test, y_onehot_test, y_test = pad_batch_2vec(imdb_processed['val'], seq_limit,
                                               imdb_processed['embed_weights'])

for epoch in range(args.max_epoch):
    i = 0
    l = 0
    correct = 0
    trials = 0
    while (i + 1) * args.bs < len(x_train):
        l_idx = i * args.bs
        r_idx = l_idx + args.bs
        x_batch = x_train[l_idx:r_idx]
        y_onehot_batch = y_onehot_train[l_idx:r_idx]
        y_batch = y_train[l_idx:r_idx]
        i += 1

        # reuse placeholders
        tx.copy_from_numpy(x_batch)
        ty.copy_from_numpy(y_onehot_batch)

        # train one batch
        out, loss = m(tx, ty)

        # save output
        l += tensor.to_numpy(loss)
        scores = tensor.to_numpy(out)
        correct += (y_batch == np.argmax(scores, 1)).sum()
        trials += len(y_batch)

    print("epoch %d loss %s; acc %.3f" % (epoch, l /
                                          (trials / args.bs), correct / trials))
    l = 0

# testing:
m.eval()

i = 0
correct = 0
trials = 0
while (i + 1) * args.bs < len(x_test):
    l_idx = i * args.bs
    r_idx = l_idx + args.bs
    x_batch = x_test[l_idx:r_idx]
    y_onehot_batch = y_onehot_test[l_idx:r_idx]
    y_batch = y_test[l_idx:r_idx]
    i += 1

    # reuse same tensors
    tx.copy_from_numpy(x_batch)
    ty.copy_from_numpy(y_onehot_batch)

    # make inference
    out = m(tx)

    # save correct predictions
    scores = tensor.to_numpy(out)
    correct += (y_batch == np.argmax(scores, 1)).sum()
    trials += len(y_batch)

print("eval acc %.3f" % (correct / trials))
