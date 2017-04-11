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
# Base unbuntu 14.04 image from nvidia/cuda
# Change tags to build with different cuda/cudnn versions:
#   FROM nvidia/cuda:8.0-cudnn5-devel-ubuntu14.04
#   FROM nvidia/cuda:7.5-cudnn5-devel-ubuntu14.04
#   FROM nvidia/cuda:7.5-cudnn4-devel-ubuntu14.04
#   FROM nvidia/cuda:7.0-cudnn4-devel-ubuntu14.04
FROM nvidia/cuda:7.5-cudnn5-devel-centos6

# install dependencies
RUN yum -y update && yum -y install git wget openssh-server cmake


# install conda, conda-build and anaconda-client
RUN wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh;
RUN bash miniconda.sh -b -p /root/miniconda
ENV PATH /root/miniconda/bin:${PATH}
RUN /root/miniconda/bin/conda config --set always_yes yes --set changeps1 no
RUN /root/miniconda/bin/conda update -q conda
RUN /root/miniconda/bin/conda install conda-build
RUN /root/miniconda/bin/conda install anaconda-client

# set environment
ENV CPLUS_INCLUDE_PATH /usr/local/lib/python2.7/dist-packages/numpy/core/include:${CPLUS_INCLUDE_PATH}
# ENV CMAKE_INCLUDE_PATH /usr/local/cuda/include:${CMAKE_INCLUDE_PATH}
# ENV CMAKE_LIBRARY_PATH /usr/local/cuda/lib64:${CMAKE_LIBRARY_PATH}

# download singa source
# RUN git clone https://github.com/apache/incubator-singa.git

# config ssh service
RUN mkdir /var/run/sshd
RUN ssh-keygen -t rsa -f /etc/ssh/ssh_host_rsa_key
RUN ssh-keygen -t dsa -f /etc/ssh/ssh_host_dsa_key

RUN sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config
RUN sed -ri 's/#UsePAM no/UsePAM no/g' /etc/ssh/sshd_config
RUN echo 'root:singa' | chpasswd

#RUN sed -i 's/PermitRootLogin without-password/PermitRootLogin yes/' /etc/ssh/sshd_config
# SSH login fix. Otherwise user is kicked off after login
#RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd

# dump environment variables into files, so that ssh can see also
RUN env | grep _ >> /etc/environment

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
