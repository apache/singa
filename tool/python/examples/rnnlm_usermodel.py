#!/usr/bin/env python
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__),'..')) 
from singa.model import *
from singa.datasets import rnnlm 

vocab_size = 3720

X_train, X_valid, workspace = rnnlm.load_data()

m = Sequential('rnnlm', sys.argv)

parw = Parameter(init='uniform', range=0.3)
m.add(Embedding(in_dim=vocab_size, out_dim=15, w_param=parw))
m.add(RNNLM(1, w_param=parw))

sgd = SGD(lr_type='fixed', step=(0,48810,56945,65080,73215), step_lr=(0.1,0.05,0.025,0.0125,0.00625))
topo = Cluster(workspace)
m.compile(loss='user_loss_rnnlm', in_dim=vocab_size, nclass=100, optimizer=sgd, cluster=topo)

m.fit(X_train, validate=X_valid, validate_steps=683, nb_epoch=81350, execpath='examples/rnnlm/rnnlm.bin')
#result = m.evaluate(X_valid, validate_steps=683, validate_freq=8135, execpath='examples/rnnlm/rnnlm.bin')
