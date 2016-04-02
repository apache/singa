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

'''
Example script of MLP model for MNIST dataset
'''
import os, sys
import numpy as np

current_path_ = os.path.dirname(__file__)
singa_root_=os.path.abspath(os.path.join(current_path_,'../../..'))
sys.path.append(os.path.join(singa_root_,'tool','python'))

from singa.driver import Driver
from singa.layer import *
from singa.model import *

def swap32(x):
    return (((x << 24) & 0xFF000000) |
            ((x <<  8) & 0x00FF0000) |
            ((x >>  8) & 0x0000FF00) |
            ((x >> 24) & 0x000000FF))

def load_dataset():
    ''' MNIST dataset
        train-images: 4 int32 headers & int8 pixels
        train-labels: 2 int32 headers & int8 labels
    '''
    print '[Load MNIST dataset]'
    fname_train_image = "examples/mnist/train-images-idx3-ubyte"
    fname_train_label = "examples/mnist/train-labels-idx1-ubyte"
    nb_header = [4, 2]

    info = swap32(np.fromfile(fname_train_image, dtype=np.uint32, count=nb_header[0]))
    nb_samples = info[1] 
    shape = (info[2],info[3])
    
    x = np.fromfile(fname_train_image, dtype=np.uint8)
    x = x[np.dtype(np.int32).itemsize*nb_header[0]:] # skip header
    x = x.reshape(nb_samples, shape[0]*shape[1]) 
    print '   data x:', x.shape
    y = np.fromfile(fname_train_label, dtype=np.uint8)
    y = y[np.dtype(np.int32).itemsize*nb_header[1]:] # skip header
    y = y.reshape(nb_samples, 1) 
    print '  label y:', y.shape

    return x, y

#-------------------------------------------------------------------
print '[Layer registration/declaration]'
d = Driver()
d.Init(sys.argv)

input = ImageInput(28, 28)
label = LabelInput()

nn = []
nn.append(input)
nn.append(Dense(2500, init='uniform'))
nn.append(Activation('stanh'))
nn.append(Dense(2000, init='uniform'))
nn.append(Activation('stanh'))
nn.append(Dense(1500, init='uniform'))
nn.append(Activation('stanh'))
nn.append(Dense(1000, init='uniform'))
nn.append(Activation('stanh'))
nn.append(Dense(500, init='uniform'))
nn.append(Activation('stanh'))
nn.append(Dense(10, init='uniform'))
loss = Loss('softmaxloss')

# updater
sgd = SGD(lr=0.001, lr_type='step')

#-------------------------------------------------------------------
batchsize = 64 
disp_freq = 10

x, y = load_dataset()

print '[Start training]'
for i in range(x.shape[0] / batchsize):
    xb, yb = x[i*batchsize:(i+1)*batchsize,:], y[i*batchsize:(i+1)*batchsize,:]
    nn[0].Feed(xb)
    label.Feed(yb)
    for h in range(1, len(nn)):
        nn[h].ComputeFeature(nn[h-1])
    loss.ComputeFeature(nn[-1], label)
    if (i+1)%disp_freq == 0:
        print '  Step {:>3}: '.format(i+1),
        loss.display()

    loss.ComputeGradient()
    for h in range(len(nn)-1, 0, -1):
        nn[h].ComputeGradient()
        sgd.Update(i+1, nn[h])

