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
This module pre-defines initial value for fields
'''

def get_init_values(identifier, **kwargs):
    '''
    This method returns field, a set of key-value pairs, that
    key is specified by identifier and values are initialized.
    '''

    field = {}

    if identifier == 'none':
        return

    if identifier == 'uniform':
        scale = kwargs['scale'] if 'scale' in kwargs else 0.05
        names = ['low', 'high']
        values = [-scale, scale]

    elif identifier == 'constant':
        names = ['value']
        values = [0]

    elif identifier == 'gaussian':
        names = ['mean', 'std']
        values = [0, 0.01]

    elif identifier == 'conv2d':
        names = ['stride', 'pad']
        values = [1, 0]

    elif identifier == 'lrn2d':
        names = ['alpha', 'beta', 'knorm']
        values = [1, 0.75, 1]

    elif identifier == 'dropout':
        names = ['ratio']
        values = [0.5]

    for i in range(len(names)):
        field[names[i]] = kwargs[names[i]] if names[i] in kwargs else values[i]

    return field
