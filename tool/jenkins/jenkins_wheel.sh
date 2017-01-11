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

# This script is used by Jenkins to compile and distribute SINGA wheel file.

echo Compile, test and distribute PySINGA...
echo parameters: $1
echo workspace: `pwd`
echo OS version: `cat /etc/issue`
echo kernal version: `uname -a`
echo CUDA version: $CUDA_VERSION
echo CUDNN version: $CUDNN_VERSION
echo OS name: $OS_VERSION
COMMIT=`git rev-parse --short HEAD`
echo COMMIT HASH: $COMMIT
# set parameters
CUDA="OFF"
CUDNN="OFF"
FOLDER=$BUILD_NUMBER/$COMMIT/$OS_VERSION-cpp
if [ $1 = "CUDA" ]; then
  CUDA="ON"
  CUDNN="ON"
  FOLDER=$BUILD_NUMBER/$COMMIT/$OS_VERSION-cuda$CUDA_VERSION-cudnn$CUDNN_VERSION
fi
echo wheel file folder: build/python/dist/whl/$FOLDER

# setup env
rm -rf build
mkdir build

if [ `uname` = "Darwin" ]; then
  EXTRA_ARGS="-DPYTHON_LIBRARY=`python-config --prefix`/lib/libpython2.7.dylib -DPYTHON_INCLUDE_DIR=`python-config --prefix`/include/python2.7/"
fi

# compile singa c++
cd build
cmake -DUSE_CUDNN=$CUDNN -DUSE_CUDA=$CUDA -DUSE_MODULES=ON $EXTRA_ARGS ../
make
# unit test cpp code
./bin/test_singa --gtest_output=xml:./gtest.xml
# compile pysinga
cd python
python setup.py bdist_wheel
# mv whl file to a folder whose name identifies the OS, CUDA, CUDNN etc.
cd dist
mkdir -p $FOLDER
mv *.whl $FOLDER/
tar czf $BUILD_NUMBER.tar.gz $FOLDER/*

# unit test python code
cd ../../../test/python
PYTHONPATH=../../build/python/ python run.py
echo Job finished...
