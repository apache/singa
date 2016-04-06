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
singa_root_ = os.path.abspath(os.path.join(current_path_,'../../..'))
sys.path.append(os.path.join(singa_root_,'tool','python'))

from singa.driver import Driver
from singa.layer import *
from singa.model import *


'''
CIFAR10 dataset can be downloaded at [https://www.cs.toronto.edu/~kriz/cifar.html]
- please specify dataset_dir
'''
dataset_dir_ = singa_root_ + "/tool/python/examples/datasets/cifar-10-batches-py"
mean_image = None

def unpickle(file):
    ''' This method loads dataset provided at CIFAR10 website
        See [https://www.cs.toronto.edu/~kriz/cifar.html] for more details
    '''
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def compute_mean_image():
    ''' This is a sample script to cmopute the average image
        of all samples in 5 dataset of cifar10
    '''
    mean = None
    nb_samples_total = 0
    for did in range(1,6):
        fname_train_data = dataset_dir_ + "/data_batch_{}".format(did)
        cifar10 = unpickle(fname_train_data)
        image = cifar10['data'].astype(dtype=np.uint8)
        if did > 1:
            image = np.vstack((image, image))
    return np.average(image, axis=0)

def load_dataset(did=1):
    ''' CIFAR10 dataset includes
        5 binary dataset, each contains 10000 images
        1 row (1 image) includes 1 label & 3072 pixels
        3072 pixels are  3 channels of a 32x32 image
    '''
    assert mean_image is not None, 'mean_image is required'
    print '[Load CIFAR10 dataset {}]'.format(did)
    fname_train_data = dataset_dir_ + "/data_batch_{}".format(did)
    cifar10 = unpickle(fname_train_data)
    image = cifar10['data'].astype(dtype=np.uint8)
    image = image - mean_image
    print '  image x:', image.shape
    label = np.asarray(cifar10['labels'], dtype=np.uint8)
    label = label.reshape(label.size, 1)
    print '  label y:', label.shape
    return image, label

#-------------------------------------------------------------------
mean_image = compute_mean_image()
# mean_image = np.fromfile('tool/python/examples/datasets/cifar10_mean_image')

print '[Layer registration/declaration]'
d = Driver()
d.Init(sys.argv)

input = ImageInput(32, 32, 3) # image width, height, channel
label = LabelInput()

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

print '[Start training]'
for dataset_id in range(train_step / batchsize):

    x, y = load_dataset(dataset_id%5+1)

    for i in range(x.shape[0] / batchsize):
        xb, yb = x[i*batchsize:(i+1)*batchsize,:], y[i*batchsize:(i+1)*batchsize,:]
        nn[0].Feed(xb)
        label.Feed(yb)
        for h in range(1, len(nn)):
            nn[h].ComputeFeature(nn[h-1])
        loss.ComputeFeature(nn[-1], label)
        if (i+1)%disp_freq == 0:
            print '  Step {:>3}: '.format(i+1 + dataset_id*(x.shape[0]/batchsize)),
            loss.display()

        loss.ComputeGradient()
        for h in range(len(nn)-1, 0, -1):
            nn[h].ComputeGradient()
            sgd.Update(i+1, nn[h])
