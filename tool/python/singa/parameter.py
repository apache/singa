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
This script includes Parameter class and a method, named set_param_field
that users can configure Param and ParamGen protos.
'''

from singa.initializations import get_init_values
from singa.utils.utility import setval, generate_name
from singa.utils.message import *
from google.protobuf import text_format


class Parameter(object):

    def __init__(self, **kwargs):
        '''
	optional
	  **kwargs
	    name  = (string) // parameter name
	    lr    = (float)  // learning rate multiplier
	    wd    = (float)  // weight decay multiplier
	    init  = (string) // init type {'constant','uniform','gaussian'}
	    value = (int)    // value for 'constant'
	    scale = (float)  // [low=-scale, high=scale] for 'uniform'
	    low   = (float)  // low value   for 'uniform'
	    high  = (float)  // high value  for 'uniform'
	    mean  = (float)  // mean for 'gaussian'
	    std   = (float)  // std  for 'gaussian'
	'''
        fields = {'lr_scale' : kwargs['lr'] if 'lr' in kwargs else 1,
                  'wd_scale' : kwargs['wd'] if 'wd' in kwargs else 1
                 }
        self.param = Message('Param', **fields).proto

        if not 'name' in kwargs:
            setval(self.param, name=generate_name('param', 1))
        else:
            pname = kwargs['name']
            # parameter name for RBM
            if 'level' in kwargs:
                pname += str(kwargs['level'])
                if pname[0] == 'b':
                    pname += '2'
            setval(self.param, name=pname)

        if 'share_from' in kwargs:
            setval(self.param, share_from=kwargs['share_from'])

        if 'init' in kwargs:
            init_values = get_init_values(kwargs['init'], **kwargs)
            if not kwargs['init'] == 'none':
                pgen = Message('ParamGen', type=enumInitMethod(kwargs['init']),
                               **init_values)
                del kwargs['init']
                setval(self.param, init=pgen.proto)
        else: # default: uniform
            pgen = Message('ParamGen', type=enumInitMethod('uniform'))
            setval(self.param, init=pgen.proto)

    def update(self, **fields):
        setval(self.param, **fields)
        setval(self.param.init, **fields)


def set_param_field(param, pname, changename=False, withnumber=True, **kwargs):
    '''
      param      = (ParamProto)
      pname      = (string)     // 'w' for wiehgt, or 'b' for bias
      changename = (bool)       // update parameter name if True
      withnumber = (bool)       // add layer number if True
      **kwargs
        w_lr = (float) // learning rate multiplier for weight, used to
                       // scale the learning rate when updating parameters.
        w_wd = (float) // weight decay multiplier for weight, used to
                       // scale the weight decay when updating parameters.
        b_lr = (float) // learning rate multiplier for bias 
        b_wd = (float) // weight decay multiplier for bias
    '''
    assert pname == 'w' or pname == 'b', 'pname should be w or b'

    lr_ = param.lr_scale
    wd_ = param.wd_scale
    initkv = {}

    if pname == 'w':
        if 'w_lr' in kwargs:
            lr_ = kwargs['w_lr']
            del kwargs['w_lr']
        if 'w_wd' in kwargs:
            wd_ = kwargs['w_wd']
            del kwargs['w_wd']
        for key, val in kwargs.items():
            if key.startswith('w_'):
                initkv[key[2:]] = val

    elif pname == 'b':
        if 'b_lr' in kwargs:
            lr_ = kwargs['b_lr']
            del kwargs['b_lr']
        if 'b_wd' in kwargs:
            wd_ = kwargs['b_wd']
            del kwargs['b_wd']
        for key, val in kwargs.items():
            if key.startswith('b_'):
                initkv[key[2:]] = val

    field = {'lr_scale' : lr_, 'wd_scale' : wd_}

    # Set/update parameter fields
    if param.name.startswith('param') or changename == True:
        if 'level' in kwargs:  # parameter name for RBM
            pname += str(kwargs['level'])
        setval(param, name=generate_name(pname, withnumber=withnumber), **field)
    else:
        setval(param, **field)

    # Set/update parameter init fields
    setval(param.init, **initkv)
