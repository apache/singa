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

import os
import sys
import string
import pb2.job_pb2 as job_pb2
import singa.driver as driver
from google.protobuf.text_format import Merge

if __name__ == '__main__':
    """Invoke the training program using this python script.
    ./bin/singa-run.sh -exec tool/python/singa.py -conf examples/cifar10/job.conf
    """
 
    i = sys.argv.index('-conf')
    s = open(sys.argv[i+1], 'r').read()
    s = str(s)
    j = job_pb2.JobProto()
    Merge(s, j)
    b = j.SerializeToString()
    d = driver.Driver()
    d.InitLog(sys.argv[0])
    d.Init(sys.argv)
    d.Train(False, b)
    #d.Test(b)
