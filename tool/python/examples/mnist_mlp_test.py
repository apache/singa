#!/usr/bin/env python
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'..')) 
from singa.model import * 
from singa.datasets import mnist 

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
