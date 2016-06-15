#!/usr/bin/env bash
#/**
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

SINGA_ROOT=/home/chonho/incubator-singa
SINGA_SRC=${SINGA_ROOT}/src
SRC_CC=(${SINGA_SRC}/core/tensor/tensor.cc \
        ${SINGA_SRC}/core/device/device.cc
       )
USR_LOCAL=/home/chonho/local

swig -c++ -python -I../../include singa.i

g++ -fPIC ${SRC_CC[@]} singa_wrap.cxx -shared -o _singa.so \
    -L${USR_LOCAL}/lib -lprotobuf -Wl,-rpath=${USR_LOCAL}/lib \
    -L../../lib -lsinga_core -lsinga_model -lsinga_utils -Wl,-rpath=../../lib \
    -std=c++11 \
    -I../.. \
    -I../../include \
    -I${SINGA_SRC} \
    -I${USR_LOCAL}/include \
    -I${USR_LOCAL}/cudnn/include \
    -I/usr/include/python2.7 \
    -I/usr/local/cuda-7.0/include
