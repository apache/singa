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
Example script of CNN model for CIFAR10 dataset
'''
import os, sys
import numpy as np

current_path_ = os.path.dirname(__file__)
singa_root_=os.path.abspath(os.path.join(current_path_,'../../..'))
sys.path.append(os.path.join(singa_root_,'tool','python'))

from singa.driver import Driver
from singa.layer import *
from singa.model import *
from singa.utils.utility import swap32

fname_mean_image = 'tool/python/examples/datasets/cifar10_mean_image'
mean_image = np.fromfile(fname_mean_image)

def load_dataset(did=1):
    ''' CIFAR10 dataset
        5 binary dataset, each contains 10000 images
        1 row (1 image) includes 1 label & 3072 pixels
        3072 pixels are  3 channels of a 32x32 image
    '''
    print '[Load CIFAR10 dataset]', did
    dataset_dir_ = singa_root_ + "/examples/cifar10/cifar-10-batches-bin"
    fname_train_data = dataset_dir_ + "/data_batch_{}.bin".format(did)
    
    nb_samples = 10000
    nb_pixels = 3 * 32 * 32  
    d = np.fromfile(fname_train_data, dtype=np.uint8)
    d = d.reshape(nb_samples, nb_pixels + 1) # +1 for label
    x = d[:, 1:] 
    x = x - mean_image
    print '   data x:', x.shape
    y = d[:, 0]
    y = y.reshape(nb_samples, 1) 
    print '  label y:', y.shape
    return x, y

def get_labellist():
    dataset_dir_ = singa_root_ + "/examples/cifar10/cifar-10-batches-bin"
    fname_label_list = dataset_dir_ + "/batches.meta.txt"
    label_list_ = np.genfromtxt(fname_label_list, dtype=str)
    return label_list_

#-------------------------------------------------------------------
print '[Layer registration/declaration]'
d = Driver()
d.Init(sys.argv)

input = Dummy()
label = Dummy()

nn = []
nn.append(input)
nn.append(Convolution2D(32, 5, 1, 2, w_std=0.0001, b_lr=2))
nn.append(MaxPooling2D(pool_size=(3,3), stride=2))
nn.append(Activation('relu'))
nn.append(LRN2D(3, alpha=0.00005, beta=0.75))
nn.append(Convolution2D(32, 5, 1, 2, b_lr=2))
nn.append(Activation('relu'))
nn.append(AvgPooling2D(pool_size=(3,3), stride=2))
nn.append(LRN2D(3, alpha=0.00005, beta=0.75))
nn.append(Convolution2D(64, 5, 1, 2))
nn.append(Activation('relu'))
nn.append(AvgPooling2D(pool_size=(3,3), stride=2))
nn.append(Dense(10, w_wd=250, b_lr=2, b_wd=0))
loss = Loss('softmaxloss')

# updater
sgd = SGD(decay=0.004, momentum=0.9, lr_type='manual', step=(0,60000,65000), step_lr=(0.001,0.0001,0.00001))

#-------------------------------------------------------------------
batchsize = 100 
disp_freq = 50
train_step = 1000

for dataset_id in range(train_step / batchsize):

    x, y = load_dataset(dataset_id%5+1)

    print '[Start training]'
    for i in range(x.shape[0] / batchsize):
        xb, yb = x[i*batchsize:(i+1)*batchsize,:], y[i*batchsize:(i+1)*batchsize,:]
        nn[0].Feed(xb, 3, 0)
        label.Feed(yb, 1, 1)
        for h in range(1, len(nn)):
            nn[h].ComputeFeature(nn[h-1])
        loss.ComputeFeature(nn[-1], label)
        if (i+1)%disp_freq == 0:
            print '  Step {:>3}: '.format(i+1 + dataset_id*(x.shape[0]/batchsize)),
            loss.display()
        loss.ComputeGradient(i+1, sgd)
