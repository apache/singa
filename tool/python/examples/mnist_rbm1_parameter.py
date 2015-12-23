#!/usr/bin/env python
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'..')) 
from singa.model import *
from singa.datasets import mnist 

rbmid = 1
pvalues = {'batchsize' : 100, 'shape' : 784, 'std_value' : 255}
X_train, X_test, workspace = mnist.load_data(
            workspace = 'examples/rbm/rbm1',
            nb_rbm = rbmid,
            checkpoint_steps = 6000,
            **pvalues)

m = Energy('rbm'+str(rbmid), sys.argv)

parw = Parameter(init='gaussian', mean=0, std=0.1)
parb = Parameter(wd=0, init='constant', value=0)
m.add(RBM(1000, w_param=parw, b_param=parb)) 

sgd = SGD(lr=0.1, decay=0.0002, momentum=0.8)
topo = Cluster(workspace)
m.compile(optimizer=sgd, cluster=topo)
m.fit(X_train, alg='cd', nb_epoch=6000)
#result = m.evaluate(X_test, test_steps=100, test_freq=500)

