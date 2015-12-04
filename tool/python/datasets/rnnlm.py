#!/usr/bin/env python
from model import *

def load_data(
         workspace = 'examples/rnnlm',
         backend = 'kvfile',
         max_window = 10
      ):

  path_train = workspace + '/train_data.bin'
  path_valid = workspace + '/valid_data.bin'
  path_test  = workspace + '/test_data.bin'


  data_train = Data(load='kData', phase='train', path=path_train, backend=backend, max_window=max_window)

  data_valid = Data(load='kData', phase='val', path=path_valid, max_window=max_window)

  #store = Store(path=path_test, backend=backen)
  #data_test = Data(load='recordinput', phase='test', conf=store)

  return data_train, data_valid, workspace

