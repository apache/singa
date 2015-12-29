#!/usr/bin/env python
from singa.model import *

def load_data(
         workspace = None, 
         backend = 'kvfile',
         batchsize = 64,
         random = 5000,
         shape = (3, 32, 32),
         std = 127.5,
         mean = 127.5
      ):

  # using cifar10 dataset
  data_dir = 'examples/cifar10'
  path_train = data_dir + '/train_data.bin'
  path_test  = data_dir + '/test_data.bin'
  path_mean  = data_dir + '/image_mean.bin'
  if workspace == None: workspace = data_dir

  store = Store(path=path_train, mean_file=path_mean, backend=backend,
              random_skip=random, batchsize=batchsize,
              shape=shape) 

  data_train = Data(load='recordinput', phase='train', conf=store)

  store = Store(path=path_test, mean_file=path_mean, backend=backend,
              batchsize=batchsize,
              shape=shape) 

  data_test = Data(load='recordinput', phase='test', conf=store)

  return data_train, data_test, workspace

