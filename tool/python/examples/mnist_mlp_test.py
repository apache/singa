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

# Sample parameter values for Mnist MLP example
pvalues = {'batchsize' : 64, 'shape' : 784,
           'std_value' : 127.5, 'mean_value' : 127.5}
X_train, X_test, workspace = mnist.load_data(**pvalues)

m = Sequential('mlp', argv=sys.argv)

m.add(Dense(2500, init='uniform', activation='tanh'))
m.add(Dense(2000, init='uniform', activation='tanh'))
m.add(Dense(1500, init='uniform', activation='tanh'))
m.add(Dense(1000, init='uniform', activation='tanh'))
m.add(Dense(500,  init='uniform', activation='tanh'))
m.add(Dense(10, init='uniform', activation='softmax'))

sgd = SGD(lr=0.001, lr_type='step')
topo = Cluster(workspace)
m.compile(loss='categorical_crossentropy', optimizer=sgd, cluster=topo)

''' For doing test only, normally users sets checkpoint path
    e.g., assume that checkpoint exists by
          m.fit(X_train, nb_epoch=100, checkpoint_freq=100)
'''
path = workspace+'/checkpoint/step100-worker0'
result = m.evaluate(X_test, batch_size=100, test_steps=100, checkpoint_path=path)
