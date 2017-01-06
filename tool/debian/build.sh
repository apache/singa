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

# execute this script at SINGA_ROOT
BUILD="OFF"
CUDA="OFF"
CUDNN="OFF"
MODULES="OFF"
PYTHON="OFF"

echo "OS version: " $OS_VERSION
COMMIT=`git rev-parse --short HEAD`
echo "Commit ID: " $COMMIT

while [[ $# -gt 0 ]]
do
key="$1"
case $key in
  -b|--build)
  BUILD="ON"
  shift
  ;;
  -m|--modules)
  MODULES="ON"
  shift
  ;;
  -c|--cuda|--CUDA)
  CUDA="ON"
  CUDNN="ON"
  shift
  ;;
  -p|--python)
  PYTHON="ON"
  shift
  ;;
  -h|--help)
  echo "Usage (execute at SINGA_ROOT folder): build.sh -b|--build -m|--modules -c|--cuda -p|--python"
  echo "      build: compile SINGA; if not set, we assume SINGA is compiled in ROOT/build/"
  echo "      modules: if set, protobuf and openblas are linked statically, hence they are removed from 'depends' in control file"
  echo "      cuda: if set, SINGA is compiled with CUDA and CUDNN"
  echo "      python: if set, package pysinga together"
  exit 0
  ;;
  *)
  echo "WRONG argument:" $key
# for jenkins
  shift
  ;;
esac
done

# get singa version
SINGA_VERSION=`grep "PACKAGE_VERSION" CMakeLists.txt |sed 's/SET(PACKAGE_VERSION\s*\"\([0-9]\.[0-9]\.[0-9]\)\"\s*)/\1/'`
echo "SINGA version: " $SINGA_VERSION


# compile singa
if [ $BUILD = "ON" ]; then
  rm -rf build
  mkdir build
  cd build
  cmake -DUSE_MODULES=$MODULES -DUSE_CUDA=$CUDA -DUSE_CUDNN=$CUDNN -DUSE_PYTHON=$PYTHON ..
  make
  cd ..
fi

# create the folder for the package
FOLDER_PREFIX="singa"
if [ -n "$BUILD_ID" ]; then
  FOLDER_PREFIX=$BUILD_ID/$COMMIT/$OS_VERSION
fi

FOLDER=$FOLDER_PREFIX-cpp
if [ $CUDA = "ON" ]; then
  FOLDER=$FOLDER_PREFIX-cuda$CUDA_VERSION-cudnn$CUDNN_VERSION
fi

if [ $PYTHON = "ON" ]
then
  FOLDER=$FOLDER/python-singa
else
  FOLDER=$FOLDER/singa
fi

echo "Path: " build/debian/$FOLDER
mkdir -p build/debian/$FOLDER

if [ $PYTHON = "ON" ]
then
  cp -r tool/debian/python-singa/* build/debian/$FOLDER/
else
  cp -r tool/debian/singa/* build/debian/$FOLDER/
fi

# remove unnecessary dependencies
if [ $MODULES = "ON" ]; then
  sed -i 's/<libopenblas-dev\>,*//' build/debian/$FOLDER/DEBIAN/control
  sed -i 's/<libprotobuf-dev\>,*//' build/debian/$FOLDER/DEBIAN/control
  sed -i 's/<protobuf-compiler\>,*//' build/debian/$FOLDER/DEBIAN/control
fi

# copy cpp and cuda files
ICL_FOLDER=build/debian/$FOLDER/usr/local/include
mkdir -p ${ICL_FOLDER}
cp -r include/singa ${ICL_FOLDER}/
cp -r build/include/singa/proto ${ICL_FOLDER}/singa
cp build/include/singa/singa_config.h ${ICL_FOLDER}/singa

LIB_FOLDER=build/debian/$FOLDER/usr/local/lib
mkdir -p ${LIB_FOLDER}
cp build/lib/libsinga.so ${LIB_FOLDER}

mkdir -p build/debian/$FOLDER/usr/share/doc/singa
cp LICENSE build/debian/$FOLDER/usr/share/doc/singa

# copy pysinga files
if [ $PYTHON = "ON" ]; then
  PY_FOLDER=build/debian/$FOLDER/usr/local/lib/singa
  mkdir -p $PY_FOLDER
  cp -r python/singa $PY_FOLDER/
  cp -r python/rafiki $PY_FOLDER/

  cp    build/python/setup.py $PY_FOLDER/
  cp    build/python/singa/singa_wrap.py $PY_FOLDER/singa/
  cp    build/python/singa/_singa_wrap.so $PY_FOLDER/singa/
  cp -r build/python/singa/proto $PY_FOLDER/singa/
fi
# change SINGA version in the control file
sed -i "s/\(Version: [0-9]\.[0-9]\.[0-9]\)/Version: $SINGA_VERSION/" build/debian/$FOLDER/DEBIAN/control
SIZE=`du -s build/debian/$FOLDER|cut -f 1`
# change the Size
sed -i "s/\(Installed-Size: [1-9][0-9]*\)/Installed-Size: $SIZE/" build/debian/$FOLDER/DEBIAN/control

dpkg -b build/debian/$FOLDER/

cd build/debian
tar czf $BUILD_ID.tar.gz $FOLDER.deb
exit 0
