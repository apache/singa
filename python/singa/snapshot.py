# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# =============================================================================
'''
This script includes io::snapshot class and its methods.

Note: This module is depreated. Please use the model module for 
checkpoing and restore.

Example usages::

    from singa import snapshot

    sn1 = snapshot.Snapshot('param', False)
    params = sn1.read()  # read all params as a dictionary

    sn2 = snapshot.Snapshot('param_new', False)
    for k, v in params.iteritems():
        sn2.write(k, v)
'''
from __future__ import absolute_import

from builtins import object
from . import singa_wrap as singa
from . import tensor


class Snapshot(object):
    ''' Class and member functions for singa::Snapshot.

    '''

    def __init__(self, f, mode, buffer_size=10):
        '''Snapshot constructor given file name and R/W mode.

        Args:
            file (string): snapshot file name.
            mode (boolean): True for write, False for read
            buffer_size (int): Buffer size (in MB), default is 10
        '''
        self.snapshot = singa.Snapshot(f.encode(), mode, buffer_size)

    def write(self, param_name, param_val):
        '''Call Write method to write a parameter

        Args:
            param_name (string): name of the parameter
            param_val (Tensor): value tensor of the parameter
        '''
        self.snapshot.Write(param_name.encode(), param_val.data)

    def read(self):
        '''Call read method to load all (param_name, param_val)

        Returns:
            a dict of (parameter name, parameter Tensor)
        '''
        params = {}
        p = self.snapshot.Read()
        for (param_name, param_val) in p:
            # print(param_name)
            params[param_name] = tensor.from_raw_tensor(param_val)
        return params
