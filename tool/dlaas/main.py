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

import os, sys
import numpy as np

current_path_ = os.path.dirname(__file__)
singa_root_=os.path.abspath(os.path.join(current_path_,'../..'))
sys.path.append(os.path.join(singa_root_,'thirdparty','protobuf-2.6.0','python'))
sys.path.append(os.path.join(singa_root_,'tool','python'))

from model import neuralnet, updater
from singa.driver import Driver
from singa.layer import *
from singa.model import save_model_parameter, load_model_parameter 
from singa.utils.utility import swap32

from PIL import Image
import glob,random, shutil, time
from flask import Flask, request, redirect, url_for
from singa.utils import kvstore, imgtool
app = Flask(__name__)

def train(batchsize,disp_freq,check_freq,train_step,workspace,checkpoint=None):
    print '[Layer registration/declaration]'
    # TODO change layer registration methods
    d = Driver()
    d.Init(sys.argv)

    print '[Start training]'

    #if need to load checkpoint
    if checkpoint:
        load_model_parameter(workspace+checkpoint, neuralnet, batchsize)
   
    for i in range(0,train_step):       
    
        for h in range(len(neuralnet)):
            #Fetch data for input layer
            if neuralnet[h].layer.type==kDummy:
                neuralnet[h].FetchData(batchsize)
            else:
                neuralnet[h].ComputeFeature()
    
        neuralnet[h].ComputeGradient(i+1, updater)
    
        if (i+1)%disp_freq == 0:
            print '  Step {:>3}: '.format(i+1),
            neuralnet[h].display()
    
        if (i+1)%check_freq == 0:   
            save_model_parameter(i+1, workspace, neuralnet)


    print '[Finish training]'
    

def product(workspace,checkpoint):
    
    print '[Layer registration/declaration]'
    # TODO change layer registration methods
    d = Driver()
    d.Init(sys.argv)

    load_model_parameter(workspace+checkpoint, neuralnet,1) 
   
    app.debug = True
    app.run(host='0.0.0.0', port=80)


@app.route("/")
def index():
    return "Hello World! This is SINGA DLAAS! Please send post request with image=file to '/predict' "

def allowed_file(filename):
    allowd_extensions_ = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in allowd_extensions_

@app.route('/predict', methods=['POST'])
def predict():
    size_=(32,32)
    pixel_length_=3*size_[0]*size_[1]
    label_num_=10
    if request.method == 'POST':
        file = request.files['image']
        if file and allowed_file(file.filename):
            im = Image.open(file).convert("RGB")
            im = imgtool.resize_to_center(im,size_)
            pixel = floatVector(pixel_length_) 
            byteArray = imgtool.toBin(im,size_)
            data = np.frombuffer(byteArray, dtype=np.uint8)
            data = data.reshape(1, pixel_length_)
            #dummy data Layer
            shape = intVector(4)
            shape[0]=1
            shape[1]=3
            shape[2]=size_[0]
            shape[3]=size_[1]
            
            for h in range(len(neuralnet)):
            #Fetch data for input layer
                if neuralnet[h].is_datalayer:
                    if not neuralnet[h].is_label:
                        neuralnet[h].Feed(data,3)
                    else:
                        neuralnet[h].FetchData(1)
                else:
                    neuralnet[h].ComputeFeature()
    
            #get result
            #data = neuralnet[-1].get_singalayer().data(neuralnet[-1].get_singalayer())    
            #prop =floatArray_frompointer(data.mutable_cpu_data())
            prop = neuralnet[-1].GetData()
            print prop
            result=[]
            for i in range(label_num_):
                result.append((i,prop[i])) 
        
            result.sort(key=lambda tup: tup[1], reverse=True)
            print result
            response="" 
            for r in result:
                response+=str(r[0])+":"+str(r[1]) 
           
            return response 
    return "error"

    
if __name__=='__main__':
   
    if sys.argv[1]=="train":
        if len(sys.argv) < 6:
            print "argv should be more than 6"
            exit()
        if len(sys.argv) > 6:
            checkpoint = sys.argv[6]
        else:
            checkpoint = None
        #training
        train(
          batchsize = int(sys.argv[2]), 
          disp_freq = int(sys.argv[3]),
          check_freq = int(sys.argv[4]), 
          train_step = int(sys.argv[5]),
          workspace = '/workspace',
          checkpoint = checkpoint,
          )
    else:
        if len(sys.argv) < 3:
            print "argv should be more than 2"
            exit()
        checkpoint = sys.argv[2]
        product(
          workspace = '/workspace',
          checkpoint = checkpoint 
        )

