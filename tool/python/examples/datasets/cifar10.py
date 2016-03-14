#!/usr/bin/env python

#/************************************************************
#*
#* Licensed to the Apache Software Foundation (ASF) under one
#* or more contributor license agreements.  See the NOTICE file
#* distributed with this work for additional information
#* regarding copyright ownership.  The ASF licenses this file
#* to you under the Apache License, Version 2.0 (the
#* "License"); you may not use this file except in compliance
#* with the License.  You may obtain a copy of the License at
#*
#*   http://www.apache.org/licenses/LICENSE-2.0
#*
#* Unless required by applicable law or agreed to in writing,
#* software distributed under the License is distributed on an
#* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
#* KIND, either express or implied.  See the License for the
#* specific language governing permissions and limitations
#* under the License.
#*
#*************************************************************/


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

