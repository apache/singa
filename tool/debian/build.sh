#!/bin/bash

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# singa-cpp

mkdir build
cd build
cmake ../../.. -DUSE_MODULES=ON 
make
cd ..

mkdir -p singa-cpp_1.0.0_amd64/usr/local/include
cp -r ../../include/singa singa-cpp_1.0.0_amd64/usr/local/include
cp -r build/include/singa/proto singa-cpp_1.0.0_amd64/usr/local/include/singa
cp build/include/singa/singa_config.h singa-cpp_1.0.0_amd64/usr/local/include/singa

mkdir -p singa-cpp_1.0.0_amd64/usr/local/lib
cp build/lib/libsinga.so singa-cpp_1.0.0_amd64/usr/local/lib

mkdir -p singa-cpp_1.0.0_amd64/usr/share/doc/singa
cp ../../LICENSE singa-cpp_1.0.0_amd64/usr/share/doc/singa/

dpkg -b singa-cpp_1.0.0_amd64

rm -rf ./build


# singa-python

mkdir build
cd build
cmake ../../.. -DUSE_PYTHON=ON -DUSE_MODULES=ON 
make
cd python
make
cd ../..

mkdir -p singa-python_1.0.0_amd64/usr/local/lib/singa/singa/
cp -r ../../python/singa singa-python_1.0.0_amd64/usr/local/lib/singa
cp -r ../../python/rafiki singa-python_1.0.0_amd64/usr/local/lib/singa

cp    build/python/setup.py singa-python_1.0.0_amd64/usr/local/lib/singa
cp    build/python/singa/singa_wrap.py singa-python_1.0.0_amd64/usr/local/lib/singa/singa
cp    build/python/singa/_singa_wrap.so singa-python_1.0.0_amd64/usr/local/lib/singa/singa/
cp -r build/python/singa/proto singa-python_1.0.0_amd64/usr/local/lib/singa/singa

mkdir -p singa-python_1.0.0_amd64/usr/share/doc/singa
cp ../../LICENSE singa-python_1.0.0_amd64/usr/share/doc/singa/

dpkg -b singa-python_1.0.0_amd64

rm -rf ./build
