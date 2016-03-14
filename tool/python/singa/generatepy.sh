#!/usr/bin/env bash
#/**
# * Copyright 2015 The Apache Software Foundation
# *
# * Licensed to the Apache Software Foundation (ASF) under one
# * or more contributor license agreements.  See the NOTICE file
# * distributed with this work for additional information
# * regarding copyright ownership.  The ASF licenses this file
# * to you under the Apache License, Version 2.0 (the
# * "License"); you may not use this file except in compliance
# * with the License.  You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */

#The following commands are only for developers adding new py apis. 
swig -c++ -python driver.i
#g++ -fPIC ../../../src/driver.cc driver_wrap.cxx -shared -o _driver.so \
# 	 -L../../../.libs/ -lsinga -DMSHADOW_USE_CUDA=0 \
#    -DMSHADOW_USE_CBLAS=1 -DMSHADOW_USE_MKL=0 -std=c++11 \
#    -I../../../include \
#    -I/usr/include/python2.7/
