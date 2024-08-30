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
from __future__ import division
from __future__ import print_function
from builtins import range

import numpy as np
import os
import sys
import gzip
import argparse
try:
    import pickle
except ImportError:
    import cPickle as pickle

from singa import opt
from singa import device
from singa import tensor


def load_train_data(file_path):
    f = gzip.open(file_path, 'rb')
    if sys.version_info.major > 2:
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
    else:
        train_set, valid_set, test_set = pickle.load(f)
    traindata = train_set[0].astype(np.float32)
    validdata = valid_set[0].astype(np.float32)
    print(traindata.shape, validdata.shape)
    return traindata, validdata


def train(data_file, use_gpu, num_epoch=10, batch_size=100):
    print('Start intialization............')
    lr = 0.0005  # Learning rate
    weight_decay = 0.0002
    hdim = 1000
    vdim = 784
    tweight = tensor.Tensor((vdim, hdim))
    tweight.gaussian(0.0, 0.1)
    tvbias = tensor.from_numpy(np.zeros(vdim, dtype=np.float32))
    thbias = tensor.from_numpy(np.zeros(hdim, dtype=np.float32))
    sgd = opt.SGD(lr=lr, momentum=0.9, weight_decay=weight_decay)

    print('Loading data ..................')
    train_x, valid_x = load_train_data(data_file)

    if use_gpu:
        dev = device.create_cuda_gpu()
    else:
        dev = device.get_default_device()

    for t in [tweight, tvbias, thbias]:
        t.to_device(dev)

    num_train_batch = train_x.shape[0] // batch_size
    print("num_train_batch = %d " % (num_train_batch))
    for epoch in range(num_epoch):
        trainerrorsum = 0.0
        print('Epoch %d' % epoch)
        for b in range(num_train_batch):
            # positive phase
            tdata = tensor.from_numpy(train_x[(b * batch_size):((b + 1) *
                                                                batch_size), :])
            tdata.to_device(dev)
            tposhidprob = tensor.mult(tdata, tweight)
            tposhidprob = tposhidprob + thbias
            tposhidprob = tensor.sigmoid(tposhidprob)
            tposhidrandom = tensor.Tensor(tposhidprob.shape, dev)
            tposhidrandom.uniform(0.0, 1.0)
            tposhidsample = tensor.gt(tposhidprob, tposhidrandom)

            # negative phase
            tnegdata = tensor.mult(tposhidsample, tweight.T())
            tnegdata = tnegdata + tvbias
            tnegdata = tensor.sigmoid(tnegdata)

            tneghidprob = tensor.mult(tnegdata, tweight)
            tneghidprob = tneghidprob + thbias
            tneghidprob = tensor.sigmoid(tneghidprob)
            error = tensor.sum(tensor.square((tdata - tnegdata)))
            trainerrorsum = error + trainerrorsum

            tgweight = tensor.mult(tnegdata.T(), tneghidprob) \
                - tensor.mult(tdata.T(), tposhidprob)
            tgvbias = tensor.sum(tnegdata, 0) - tensor.sum(tdata, 0)
            tghbias = tensor.sum(tneghidprob, 0) - tensor.sum(tposhidprob, 0)

            sgd.apply('w', tweight, tgweight)
            sgd.apply('vb', tvbias, tgvbias)
            sgd.apply('hb', thbias, tghbias)

        print('training erroraverage = %f' %
              (tensor.to_numpy(trainerrorsum) / train_x.shape[0]))

        tvaliddata = tensor.from_numpy(valid_x)
        tvaliddata.to_device(dev)
        tvalidposhidprob = tensor.mult(tvaliddata, tweight)
        tvalidposhidprob = tvalidposhidprob + thbias
        tvalidposhidprob = tensor.sigmoid(tvalidposhidprob)
        tvalidposhidrandom = tensor.Tensor(tvalidposhidprob.shape, dev)
        tvalidposhidrandom.uniform(0.0, 1.0)
        tvalidposhidsample = tensor.gt(tvalidposhidprob, tvalidposhidrandom)

        tvalidnegdata = tensor.mult(tvalidposhidsample, tweight.T())
        tvalidnegdata = tvalidnegdata + tvbias
        tvalidnegdata = tensor.sigmoid(tvalidnegdata)

        validerrorsum = tensor.sum(tensor.square((tvaliddata - tvalidnegdata)))
        print('valid erroraverage = %f' %
              (tensor.to_numpy(validerrorsum) / valid_x.shape[0]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RBM over MNIST')
    parser.add_argument('file', type=str, help='the dataset path')
    parser.add_argument('--use_gpu', action='store_true')
    args = parser.parse_args()

    assert os.path.exists(args.file), 'Pls download the MNIST dataset from' \
        'https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz'
    train(args.file, args.use_gpu)
