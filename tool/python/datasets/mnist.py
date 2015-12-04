#!/usr/bin/env python
from model import * 

def load_data(
         workspace = 'examples/mnist',
         backend = 'kvfile',
         random = 5000,
         batchsize = 64,
         shape = 784,
         std = 127.5,
         mean = 127.5
      ):

  path_train = workspace + '/train_data.bin'
  path_test  = workspace + '/test_data.bin'

  store = Store(path=path_train, backend=backend,
                random_skip=random,
                batchsize=batchsize, shape=shape,
                std_value=std, mean_value=mean)
  data_train = Data(load='recordinput', phase='train', conf=store)

  store = Store(path=path_test, backend=backend,
                batchsize=batchsize, shape=shape,
                std_value=std, mean_value=mean)
  data_test = Data(load='recordinput', phase='test', conf=store)

  return data_train, data_test, workspace

