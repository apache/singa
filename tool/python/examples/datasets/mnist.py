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
     nb_rbm = 0,  # the number of layers for RBM and Autoencoder
     checkpoint_steps = 0,
     **pvalues
   ):

  # using mnist dataset
  data_dir = 'examples/mnist'
  path_train = data_dir + '/train_data.bin'
  path_test  = data_dir + '/test_data.bin'
  if workspace == None: workspace = data_dir

  # checkpoint path to load
  checkpoint_list = None
  if checkpoint_steps > 0:
    workerid = 0
    checkpoint_list = []
    for i in range(nb_rbm-1, 0, -1):
      checkpoint_list.append('examples/rbm/rbm{0}/checkpoint/step{1}-worker{2}'.format(str(i),checkpoint_steps,workerid))

  store = Store(path=path_train, backend=backend, **pvalues)
  data_train = Data(load='recordinput', phase='train', conf=store, checkpoint=checkpoint_list)

  store = Store(path=path_test, backend=backend, **pvalues)
  data_test = Data(load='recordinput', phase='test', conf=store)

  return data_train, data_test, workspace
