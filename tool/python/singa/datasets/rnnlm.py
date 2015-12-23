#!/usr/bin/env python
from singa.model import *

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

  return data_train, data_valid, workspace

