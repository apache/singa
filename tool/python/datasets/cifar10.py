#!/usr/bin/env python
from model import *

def load_data(
         workspace = 'examples/cifar10',
         path_mean = 'examples/cifar10/image_mean.bin',
         backend = 'kvfile',
         batchsize = 64,
         random = 5000,
         shape = (3, 32, 32),
         std = 127.5,
         mean = 127.5
      ):

  path_train = workspace + '/train_data.bin'
  path_test  = workspace + '/test_data.bin'

  store = Store(path=path_train, mean_file=path_mean, backend=backend,
              random_skip=random, batchsize=batchsize,
              shape=shape) 

  data_train = Data(load='recordinput', phase='train', conf=store)

  store = Store(path=path_test, mean_file=path_mean, backend=backend,
              batchsize=batchsize,
              shape=shape) 

  data_test = Data(load='recordinput', phase='test', conf=store)

  return data_train, data_test, workspace

