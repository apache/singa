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
FROM nvidia/cuda:9.0-devel-ubuntu16.04

ENV CUDNN_VERSION 7.6.5.32

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcudnn7=$CUDNN_VERSION-1+cuda9.0 \
    libcudnn7-dev=$CUDNN_VERSION-1+cuda9.0 && \
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
    tqdm \
    wheel \
    numpy \
    setuptools \
    protobuf \
    Deprecated \
    future \
    onnx==1.6.0

# install swig > 3.0.10
RUN wget http://prdownloads.sourceforge.net/swig/swig-3.0.10.tar.gz -P /tmp/ \
    && tar zxf /tmp/swig-3.0.10.tar.gz -C /tmp/ \
    && cd /tmp/swig-3.0.10 && ./configure && make && make install

# install dnnl
RUN wget https://github.com/intel/mkl-dnn/releases/download/v1.1/dnnl_lnx_1.1.0_cpu_gomp.tgz -P /tmp/ \
    && tar zxf /tmp/dnnl_lnx_1.1.0_cpu_gomp.tgz -C /root
ENV DNNL_ROOT /root/dnnl_lnx_1.1.0_cpu_gomp/

# config ssh service
RUN mkdir /var/run/sshd \
    && echo 'root:singa' | chpasswd \
    # for ubuntu 16.04 prohibit
    && sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    # SSH login fix. Otherwise user is kicked off after login
    && sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd \
    # dump environment variables into files, so that ssh can see also
    && env | grep _ >> /etc/environment

# build nccl
RUN git clone https://github.com/NVIDIA/nccl.git $HOME/nccl \
    && cd $HOME/nccl \
    && git checkout v2.4.8-1 \
    && make -j src.build \
    && apt-get update \
    && apt install build-essential devscripts debhelper fakeroot -y \
    && make pkg.debian.build \
    && dpkg -i build/pkg/deb/libnccl2_2.4.8-1+cuda9.0_amd64.deb \
    && dpkg -i build/pkg/deb/libnccl-dev_2.4.8-1+cuda9.0_amd64.deb

# build mpiexec
RUN wget http://www.mpich.org/static/downloads/3.3.2/mpich-3.3.2.tar.gz -P $HOME \
    && cd $HOME \
    && tar xfz mpich-3.3.2.tar.gz \
    && cd mpich-3.3.2 \
    && ./configure --prefix=$HOME/mpich-3.3.2/build --disable-fortran 2>&1 | tee c.txt \
    && make 2>&1 | tee m.txt \
    && make install 2>&1 | tee mi.txt
ENV PATH=/root/mpich-3.3.2/build/bin:$PATH

# build singa
RUN git clone https://github.com/apache/singa.git $HOME/singa \
    && cd $HOME/singa \
    && git checkout dev \
    && mkdir build && cd build \
    && cmake -DENABLE_TEST=ON -DUSE_CUDA=ON -DUSE_PYTHON3=ON -DUSE_DNNL=ON -DUSE_DIST=ON ..
RUN cd $HOME/singa/build && make && make install
ENV PYTHONPATH="/root/singa/build/python/"

WORKDIR /root/singa
EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
