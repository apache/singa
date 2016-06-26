import os, sys
import numpy as np

current_path_ = os.path.dirname(__file__)
singa_root_="/usr/src/incubator-singa"
sys.path.append(os.path.join(singa_root_,'tool','python'))

from singa.driver import Driver
from singa.layer import *
from singa.model import *
from singa.utils.utility import swap32


imageData_0=ImageData(shape=[50000,3,32,32],data_path="/workspace/data/train.bin",data_type="byte",mean_path="/workspace/data/mean.bin",mean_type="float")
labelData_0=LabelData(shape=[50000,1],label_path="/workspace/data/train.label.bin",label_type="int")
convolution_0=Convolution2D(32,5,1,2,w_std=0.0001, b_lr=2,src=[imageData_0])
pooling_0=MaxPooling2D(pool_size=(3,3), stride=2,src=[convolution_0])
activation_0=Activation("relu",src=[pooling_0])
lrn_0=LRN2D(3, alpha=0.00005, beta=0.75,src=[activation_0])
convolution_1=Convolution2D(32,5,1,2, b_lr=2,src=[lrn_0])
activation_1=Activation("relu",src=[convolution_1])
pooling_1=AvgPooling2D(pool_size=(3,3), stride=2,src=[activation_1])
lrn_1=LRN2D(3, alpha=0.00005, beta=0.75,src=[pooling_1])
convolution_2=Convolution2D(64,5,1,2, b_lr=2,src=[lrn_1])
activation_2=Activation("relu",src=[convolution_2])
pooling_2=AvgPooling2D(pool_size=(3,3), stride=2,src=[activation_2])
dense_0=Dense(10, w_wd=250, b_lr=2, b_wd=0,src=[pooling_2])
loss_0=Loss("softmaxloss",src=[dense_0,labelData_0])
neuralnet = [imageData_0,labelData_0,convolution_0,pooling_0,activation_0,lrn_0,convolution_1,activation_1,pooling_1,lrn_1,convolution_2,activation_2,pooling_2,dense_0,loss_0]



#algorithm
updater = SGD(decay=0.004, momentum=0.9, lr_type='manual', step=(0,60000,65000), step_lr=(0.001,0.0001,0.00001))