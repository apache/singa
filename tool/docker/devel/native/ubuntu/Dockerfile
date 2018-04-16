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
# Base unbuntu 16.04 image
FROM ubuntu:latest

MAINTAINER incubator-singa dev@singa.incubator.apache.org

# install dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git cmake libprotobuf-dev libopenblas-dev protobuf-compiler python-dev python-pip swig wget\
    && apt-get clean && apt-get autoremove && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* \
    && pip --no-cache-dir install -U pip numpy setuptools

# set environment
ENV CPLUS_INCLUDE_PATH /usr/local/lib/python2.7/dist-packages/numpy/core/include:${CPLUS_INCLUDE_PATH}

# download singa source
RUN git clone https://github.com/apache/incubator-singa.git

# compile singa and install pysinga
RUN cd incubator-singa && mkdir build && cd build && cmake .. && make && cd python && pip install . && cd ../../ && rm -rf build
