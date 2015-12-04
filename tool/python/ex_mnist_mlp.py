#!/usr/bin/env python
from model import *
from datasets import mnist 

X_train, X_test, workspace = mnist.load_data()

m = Sequential('mlp')

par = Parameter(init='uniform', range=0.05)
m.add(Dense(2500, w_param=par, b_param=par, activation='tanh')) 
m.add(Dense(2000, w_param=par, b_param=par, activation='tanh')) 
m.add(Dense(1500, w_param=par, b_param=par, activation='tanh')) 
m.add(Dense(1000, w_param=par, b_param=par, activation='tanh')) 
m.add(Dense(500, w_param=par, b_param=par, activation='tanh')) 
m.add(Dense(10, w_param=par, b_param=par, activation='softmax')) 

sgd = SGD(lr=0.001, lr_type='step')
topo = Cluster(workspace)
m.compile(loss='categorical_crossentropy', optimizer=sgd, cluster=topo)
m.fit(X_train, train_steps=100, disp_freq=10)
result = m.evaluate(X_test, batch_size=100, test_steps=10)

#TODO---- classify/predict for new data
#label = m.predict(data_new, ...)
#-------

#print
#m.display()

