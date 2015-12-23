#!/usr/bin/env python
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'..')) 
from singa.model import *
from singa.datasets import mnist 

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

