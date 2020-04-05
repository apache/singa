#!/usr/bin/env sh
#
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
FROM ubuntu:18.04

MAINTAINER singa dev@singa.apache.org

# install dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends subversion git wget openssh-server bzip2 \
    && apt-get clean && apt-get autoremove && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/*

# install conda
RUN wget --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-py37_4.8.2-Linux-x86_64.sh -O miniconda.sh
RUN bash miniconda.sh -b -p /root/miniconda
ENV PATH /root/miniconda/bin:${PATH}
RUN conda config --set always_yes yes --set changeps1 no
RUN conda config --add channels conda-forge
RUN conda config --add channels nusdbsystem
RUN conda install -c nusdbsystem singa-cpu
RUN conda install -c conda-forge sphinx
RUN conda install -c conda-forge sphinx_rtd_theme
RUN conda install -c conda-forge recommonmark


# config ssh service
RUN mkdir /var/run/sshd \
    && echo 'root:singa' | chpasswd \
    && sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config \
    && mkdir /root/.ssh

# add conda bin path to login or non-login shell
RUN echo PATH=$PATH:/root/miniconda/bin >> /etc/profile

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
