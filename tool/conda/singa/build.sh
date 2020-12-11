# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# to compile swig api files which depdend on numpy.i
# export CPLUS_INCLUDE_PATH=`python -c "from __future__ import print_function; import numpy; print(numpy.get_include())"`:$CPLUS_INCLUDE_PATH

# to let cmake use the dependent libs installed by conda, including python
export CMAKE_PREFIX_PATH=$PREFIX:$CMAKE_PREFIX_PATH
export CMAKE_INCLUDE_PATH=$PREFIX/include:$CMAKE_INCLUDE_PATH
export CMAKE_LIBRARY_PATH=$PREFIX/lib:$CMAKE_LIBRARY_PATH


# if [ -z ${CUDA+x} ]; then
if [ -z "$CUDA" ]; then
	USE_CUDA=OFF
else
	USE_CUDA=ON
fi


if [ $DIST == "ON" ]; then
	USE_DIST=ON
else
	USE_DIST=OFF
fi

mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$PREFIX -DUSE_CUDA=$USE_CUDA \
	-DUSE_PYTHON3=ON -DUSE_DNNL=ON -DUSE_DIST=$USE_DIST -DCMAKE_OSX_SYSROOT=${CONDA_BUILD_SYSROOT} ..

make
make install
