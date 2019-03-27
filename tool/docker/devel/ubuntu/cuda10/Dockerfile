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
FROM nvidia/cuda:10.0-devel-ubuntu18.04

ENV CUDNN_VERSION 7.4.2.24
RUN apt-get update && apt-get install -y --no-install-recommends \
        libcudnn7=$CUDNN_VERSION-1+cuda10.0 \
        libcudnn7-dev=$CUDNN_VERSION-1+cuda10.0 && \
    apt-mark hold libcudnn7 && \
    rm -rf /var/lib/apt/lists/*

# install dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        autoconf \
        libtool \
        libprotobuf-dev \
        libopenblas-dev \
        libpcre3-dev \
        protobuf-compiler \
        wget \
        swig \
        openssh-server \
        python3-dev \
        python3-pip \
        python3-setuptools \
        libgoogle-glog-dev \
        sudo \
    && apt-get clean \
    && apt-get autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install -U --no-cache \
        wheel \
        numpy \
        setuptools \
        protobuf \
        future

# install cmake to correctly find Cuda 10
RUN wget https://github.com/Kitware/CMake/releases/download/v3.12.2/cmake-3.12.2.tar.gz -P /tmp/ \
    && tar zxf /tmp/cmake-3.12.2.tar.gz -C /tmp/ \
    && cd /tmp/cmake-3.12.2/ && ./bootstrap && make -j4 && make install

# install mkldnn
RUN wget https://github.com/intel/mkl-dnn/archive/v0.18.tar.gz -P /tmp/ \
    && tar zxf /tmp/v0.18.tar.gz -C /tmp/ \
    && cd /tmp/mkl-dnn-0.18/ \
    && cd scripts && ./prepare_mkl.sh && cd .. \
    && mkdir -p build && cd build && cmake .. \
    && make && make install

# config ssh service
RUN mkdir /var/run/sshd \
    && echo 'root:singa' | chpasswd \
    # for ubuntu 16.04 prohibit
    && sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    # SSH login fix. Otherwise user is kicked off after login
    && sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd \
    # dump environment variables into files, so that ssh can see also
    && env | grep _ >> /etc/environment \
    && useradd -m singa && echo 'singa:singa' | chpasswd && adduser singa sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER singa 
# build incubator singa
RUN git clone https://github.com/apache/incubator-singa.git $HOME/incubator-singa \
    && cd $HOME/incubator-singa \
    && mkdir build && cd build \
    && /usr/local/bin/cmake -DENABLE_TEST=ON -DUSE_CUDA=ON -DUSE_PYTHON3=ON -DUSE_MKLDNN=ON ..
RUN cd $HOME/incubator-singa/build && make && sudo make install

WORKDIR /home/singa/incubator-singa
EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
