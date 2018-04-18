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

# Change tags to build with different cuda/cudnn versions:
FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04

# install dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential cmake wget openssh-server ca-certificates\
    && apt-get clean && apt-get autoremove && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* 

# install conda, conda-build and anaconda-client
RUN wget --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh;
RUN bash miniconda.sh -b -p /root/miniconda
RUN /root/miniconda/bin/conda config --set always_yes yes --set changeps1 no
RUN /root/miniconda/bin/conda update -q conda
RUN /root/miniconda/bin/conda install conda-build
RUN /root/miniconda/bin/conda install anaconda-client
ENV PATH /root/miniconda/bin:${PATH}


# config ssh service
RUN mkdir /var/run/sshd
RUN echo 'root:singa' | chpasswd
# for ubuntu 14.04
# RUN sed -i 's/PermitRootLogin without-password/PermitRootLogin yes/' /etc/ssh/sshd_config
# for ubuntu 16.04 prohibit
RUN sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
# SSH login fix. Otherwise user is kicked off after login
RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

# dump environment variables into files, so that ssh can see also
RUN env | grep _ >> /etc/environment

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
