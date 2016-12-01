#!/usr/bin/env bash 
#/**
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

echo Compile, test and distribute PySINGA...
echo Parameters: $1
echo Workspace: `pwd`
echo OS env: `uname -a`
echo Cuda env: `nvcc --version`
# set parameters
CUDNN="OFF"
if [ $1 = "cudnn" ]; then
  CUDNN="ON"
fi
# setup env
rm -rf build
mkdir build
# compile singa c++
cd build
cmake -DUSE_CUDNN=$CUDNN -DUSE_CUDA=$CUDNN ../ 
make
# unit test
./bin/test_singa --gtest_output=xml:./../gtest.xml
# compile pysinga
cd python
python setup.py bdist_wheel
# rename dist
cd dist
mv singa-1.0.1-py2-none-any.whl singa-1.0.0-cp27-none-linux_x86_64.whl
echo Job finished...
