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
# 
# Change tags to build with different cuda/cudnn versions:
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# install dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        autoconf \
        libtool \
        cmake \
        libprotobuf-dev \
        libopenblas-dev \
        libpcre3-dev \
        protobuf-compiler \
        wget \
        openssh-server \
        swig \
        python3-dev \
        python3-pip \
        python3-setuptools \
    && apt-get clean \
    && apt-get autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install -U --no-cache \
        wheel \
        numpy \
        setuptools \
        protobuf \
        future \
    # install swig > 3.0.10 for ubuntu < 16.04
    # RUN wget http://prdownloads.sourceforge.net/swig/swig-3.0.10.tar.gz && \
    #     tar zxf swig-3.0.10.tar.gz && cd swig-3.0.10 && \
    #     ./configure && make && make install
    # set environment
    # ENV CMAKE_INCLUDE_PATH /usr/local/cuda/include:${CMAKE_INCLUDE_PATH}
    # ENV CMAKE_LIBRARY_PATH /usr/local/cuda/lib64:${CMAKE_LIBRARY_PATH}
    # config ssh service
    && mkdir /var/run/sshd \
    && echo 'root:singa' | chpasswd \
    # for ubuntu 14.04
    # RUN sed -i 's/PermitRootLogin without-password/PermitRootLogin yes/' /etc/ssh/sshd_config
    # for ubuntu 16.04 prohibit
    && sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    # SSH login fix. Otherwise user is kicked off after login
    && sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd \
    # dump environment variables into files, so that ssh can see also
    && env | grep _ >> /etc/environment

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
