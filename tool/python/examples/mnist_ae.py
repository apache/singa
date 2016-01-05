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


import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'..'))
from singa.model import *
from examples.datasets import mnist

# Sample parameter values for Autoencoder example
rbmid = 4
pvalues = {'batchsize' : 100, 'shape' : 784, 'std_value' : 255}
X_train, X_test, workspace = mnist.load_data(
            workspace = 'examples/rbm/autoencoder',
            nb_rbm = rbmid+1,
            checkpoint_steps = 6000,
            **pvalues)

m = Sequential('autoencoder', sys.argv)

hid_dim = [1000, 500, 250, 30]
m.add(Autoencoder(hid_dim, out_dim=784, activation='sigmoid', param_share=True))

agd = AdaGrad(lr=0.01)
topo = Cluster(workspace)
m.compile(loss='mean_squared_error', optimizer=agd, cluster=topo)
m.fit(X_train, alg='bp', nb_epoch=12200, with_test=True)
result = m.evaluate(X_test, test_steps=100, test_freq=1000)

