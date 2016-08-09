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

import cPickle
import numpy as np
import numpy.matlib
import os
import sys
import gzip, numpy


sys.path.append(os.path.join(os.path.dirname(__file__),
                             '../../build/python'))
sys.path.append(os.path.join(os.path.dirname(__file__),
                             '../../build/lib'))
sys.path.append(os.path.join(os.path.dirname(__file__),'../../build/src'))
from singa import initializer
from singa import utils
from singa import optimizer
from singa import device
from singa import tensor
from singa.proto import core_pb2



def load_train_data(dir_path):
    f = gzip.open(dir_path, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    traindata = train_set[0].astype(np.float32)
    validdata = valid_set[0].astype(np.float32)
    return traindata, validdata



def train(data_dir, num_epoch=10, batch_size=100):
    print 'Start intialization............'
    lr = 0.1   # Learning rate
    weight_decay  = 0.0002
    hdim = 1000
    vdim = 784
    opt = optimizer.SGD(momentum=0.8, weight_decay=weight_decay)
    
    shape = (vdim, hdim)
    tweight = tensor.Tensor(shape)
    initializer.gaussian(tweight, 0.0, 0.1)
    tvbias = tensor.from_numpy(np.zeros(vdim, dtype = np.float32))
    thbias = tensor.from_numpy(np.zeros(hdim, dtype = np.float32))
    opt = optimizer.SGD(momentum=0.5, weight_decay=weight_decay)

    print 'Loading data ..................'
    train_x, valid_x = load_train_data(data_dir)

    num_train_batch = train_x.shape[0]/batch_size
    print "num_train_batch = \n", num_train_batch
    for epoch in range(num_epoch):
        trainerrorsum = 0.0
        validerrorsum = 0.0
        print 'Epoch %d' % epoch
        for b in range(num_train_batch):
            # positive phase
            if b % 100 == 0:
                print "batch: \n", b

            tdata = tensor.from_numpy(train_x[ (b * batch_size): ((b + 1) * batch_size), : ])
            tposhidprob = tensor.mult(tdata, tweight)
            tposhidprob.add_row(thbias)
            tposhidprob = tensor.sigmoid(tposhidprob)
            tposhidrandom = tensor.Tensor(tposhidprob.shape)
            initializer.uniform(tposhidrandom, 0.0, 1.0)
            tposhidsample = tensor.gt(tposhidprob, tposhidrandom)
            
            # negative phase
            tnegdata = tensor.mult(tposhidsample, tweight.transpose())
            tnegdata.add_row(tvbias)
            tnegdata = tensor.sigmoid(tnegdata)

            tneghidprob = tensor.mult(tnegdata, tweight)
            tneghidprob.add_row(thbias) 
            tneghidprob = tensor.sigmoid(tneghidprob)
            trainerror = tensor.sum(tensor.eltwise_mult((tdata - tnegdata),(tdata - tnegdata)))
            trainerrorsum = trainerror + trainerrorsum
           
            tgweight = tensor.mult(tnegdata.transpose(), tneghidprob) - tensor.mult(tdata.transpose(), tposhidprob)
            tgvbias = tensor.sum(tnegdata, 0) - tensor.sum(tdata, 0)
            tghbias = tensor.sum(tneghidprob, 0) - tensor.sum(tposhidprob, 0)
            
            opt.apply_with_lr(epoch, lr / batch_size, tgweight, tweight, '')
            opt.apply_with_lr(epoch, lr / batch_size, tgvbias, tvbias, '')
            opt.apply_with_lr(epoch, lr / batch_size, tghbias, thbias, '')

        info = 'train errorsum = %f' \
            % (trainerrorsum)
        print info

        tvaliddata = tensor.from_numpy(valid_x[ :, : ])
        tvalidposhidprob = tensor.mult(tvaliddata, tweight)
        tvalidposhidprob.add_row(thbias)
        tvalidposhidprob = tensor.sigmoid(tvalidposhidprob)
        tvalidposhidrandom = tensor.Tensor(tvalidposhidprob.shape)
        initializer.uniform(tvalidposhidrandom, 0.0, 1.0)
        tvalidposhidsample = tensor.gt(tvalidposhidprob, tvalidposhidrandom)

        tvalidnegdata = tensor.mult(tvalidposhidsample, tweight.transpose())
        tvalidnegdata.add_row(tvbias)
        tvalidnegdata = tensor.sigmoid(tvalidnegdata)

        validerrorsum = tensor.sum(tensor.eltwise_mult((tvaliddata - tvalidnegdata),(tvaliddata - tvalidnegdata)))
        validinfo = 'valid errorsum = %f' \
            % (validerrorsum)
        print validinfo


if __name__ == '__main__':
    data_dir = 'mnist.pkl.gz'
    assert os.path.exists(data_dir), \
        'Pls download the mnist dataset'
    train(data_dir)
