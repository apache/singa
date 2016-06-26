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
This script includes methods to
(1) generate name of layer, parameter, etc.
(2) set field values for proto.
(3) swap bits
'''

LAYERID = 0
PARAMID = 0

def generate_name(label, option=0, withnumber=True):
    ''' This method returns name of layer or parameter with unique id.
        option: 1 to increase id number
        withnumber: True to concatenate number to name
    '''

    global LAYERID, PARAMID
    num = LAYERID
    if label == 'layer':
        if option == 1: LAYERID += 1
        num = LAYERID
    elif label == 'param':
        if option == 1: PARAMID += 1
        num = PARAMID
    else:
        if option == 1: LAYERID += 1
        num = LAYERID
        if option == 2:
            num = LAYERID+1

    if withnumber == False:
        return '{0}'.format(label)

    return '{0}{1}'.format(label, num)

def setval(proto, **kwargs):
    ''' This method sets field values for give proto.
    '''

    for key, val in kwargs.items():
        #print 'kv: ', k, ', ', v
        if hasattr(proto, key):
            flabel = proto.DESCRIPTOR.fields_by_name[key].label
            ftype = proto.DESCRIPTOR.fields_by_name[key].type

            fattr = getattr(proto, key)
            if flabel == 3: # repeated field
                if ftype == 11: # message type
                    fattr = fattr.add()
                    fattr.MergeFrom(val)
                else:
                    if type(val) == list or type(val) == tuple:
                        for i in range(len(val)):
                            fattr.append(val[i])
                    else:
                        fattr.append(val)
            else:
                if ftype == 11: # message type
                    fattr = getattr(proto, key)
                    fattr.MergeFrom(val)
                else:
                    setattr(proto, key, val)

def swap32(x):
    return (((x << 24) & 0xFF000000) |
            ((x <<  8) & 0x00FF0000) |
            ((x >>  8) & 0x0000FF00) |
            ((x >> 24) & 0x000000FF))

def blob_to_numpy(blob):
    '''TODO This method transform blob data to python numpy array 
    '''
    pass
