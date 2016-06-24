#!/usr/bin/env python

# /************************************************************
# *
# * Licensed to the Apache Software Foundation (ASF) under one
# * or more contributor license agreements.  See the NOTICE file
# * distributed with this work for additional information
# * regarding copyright ownership.  The ASF licenses this file
# * to you under the Apache License, Version 2.0 (the
# * "License"); you may not use this file except in compliance
# * with the License.  You may obtain a copy of the License at
# *
# *   http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing,
# * software distributed under the License is distributed on an
# * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# * KIND, either express or implied.  See the License for the
# * specific language governing permissions and limitations
# * under the License.
# *
# *************************************************************/

'''
This script includes Device class and its subclasses for python users
to call singa::Device and its methods
'''
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__),
                             '../../build/lib'))
sys.path.append(os.path.join(os.path.dirname(__file__),
                             '../../build/python'))
import singa


class Device(object):
    ''' Class and member functions for singa::Device
    '''

    def __init__(self, id=-1, num_executors=1, scheduler='sync', vm='gc-only',
                 device='cpu'):
        ''' id = (int)            // device ID
            num_executors = (int) // # of executors (e.g., cuda streams)
            scheduler = (string)  // identifier of scheduler type (default
                                  // scheduler run operations synchronously)
            vm = (string)         // virtual memory type (default vm only
                                  // provides garbage collection)
            (TODO) max mem size to use (in MB)
        '''
        if device == 'gpu':
            self.singa_device = singa.CudaGPU(id, num_executors, scheduler, vm)
        else:
            self.singa_device = singa.CppCPU(id, num_executors, scheduler, vm)

        self.id = id
        self.num_executors = num_executors
        self.scheduler = scheduler
        self.vm = vm

    def set_rand_seed(self, seed):
        self.singa_device.SetRandSeed(seed)

    def get_host(self):
        return self.singa_device.host()

    def get_id(self):
        return self.singa_device.id()


class CppCPU(Device):

    def __init__(self, id=-1, num_executors=1, scheduler='sync', vm='gc-only'):
        super(CppCPU, self).__init__(id, num_executors, scheduler, vm)


class CudaGPU(Device):

    def __init__(self, id=0, num_executors=1, scheduler='sync', vm='gc-only'):
        super(CudaGPU, self).__init__(id, num_executors, scheduler, vm, 'gpu')
