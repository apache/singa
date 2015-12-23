#!/usr/bin/env python
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'..')) 
from singa.model import * 
from singa.datasets import mnist 

# Sample parameter values for Mnist MLP example
pvalues = {'batchsize' : 64, 'shape' : 784,
           'random_skip' : 5000,
           'std_value' : 127.5, 'mean_value' : 127.5}
X_train, X_test, workspace = mnist.load_data(**pvalues)

m = Sequential('mlp', argv=sys.argv)

par = Parameter(init='uniform', scale=0.05)
m.add(Dense(2500, w_param=par, b_param=par, activation='tanh')) 
m.add(Dense(2000, w_param=par, b_param=par, activation='tanh')) 
m.add(Dense(1500, w_param=par, b_param=par, activation='tanh')) 
m.add(Dense(1000, w_param=par, b_param=par, activation='tanh')) 
m.add(Dense(500, w_param=par, b_param=par, activation='tanh')) 
m.add(Dense(10, w_param=par, b_param=par, activation='softmax')) 

sgd = SGD(lr=0.001, lr_type='step')
topo = Cluster(workspace)
m.compile(loss='categorical_crossentropy', optimizer=sgd, cluster=topo)

m.fit(X_train, nb_epoch=100, with_test=True)
result = m.evaluate(X_test, batch_size=100, test_steps=10)
