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

FROM ubuntu:18.04

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
    cmake \
    && apt-get clean \
    && apt-get autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install -U --no-cache \
    wheel \
    numpy \
    setuptools \
    protobuf \
    Deprecated \
    future

# install dnnl
RUN wget https://github.com/intel/mkl-dnn/releases/download/v1.1/dnnl_lnx_1.1.0_cpu_gomp.tgz -P /tmp/ \
    && tar zxf /tmp/dnnl_lnx_1.1.0_cpu_gomp.tgz -C /root
ENV DNNL_ROOT /root/dnnl_lnx_1.1.0_cpu_gomp/

# config ssh service
RUN mkdir /var/run/sshd \
    && echo 'root:singa' | chpasswd \
    && sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config \
    && mkdir /root/.ssh

# build singa
RUN git clone https://github.com/apache/singa.git $HOME/singa \
    && cd $HOME/singa \
    && mkdir build && cd build \
    && cmake -DENABLE_TEST=ON -DUSE_PYTHON3=ON -DUSE_DNNL=ON ..
RUN cd $HOME/singa/build && make && make install

WORKDIR $HOME/singa
EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]

