#!/usr/bin/env python
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'..')) 
from singa.model import *
from singa.datasets import mnist 

rbmid = 3
pvalues = {'batchsize' : 100, 'shape' : 784, 'std_value' : 255}
X_train, X_test, workspace = mnist.load_data(
            workspace = 'examples/rbm/rbm3',
            nb_rbm = rbmid,
            checkpoint_steps = 6000,
            **pvalues)

m = Energy('rbm'+str(rbmid), sys.argv)

out_dim = [1000, 500, 250]
m.add(RBM(out_dim, w_std=0.1, b_wd=0)) 

sgd = SGD(lr=0.1, decay=0.0002, momentum=0.8)
topo = Cluster(workspace)
m.compile(optimizer=sgd, cluster=topo)
m.fit(X_train, alg='cd', nb_epoch=6000)
#result = m.evaluate(X_test, test_steps=100, test_freq=500)

