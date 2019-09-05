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
FROM nvidia/cuda:9.0-devel-ubuntu16.04


# install dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        git \
        build-essential \
        cmake \
        wget \
        openssh-server \
        ca-certificates \
    && apt-get clean \
    && apt-get autoremove \
    && apt-get autoclean \
    && rm -rf /var/lib/apt/lists/* \
    #
    # install conda, conda-build and anaconda-client
    #
    && wget --no-check-certificate https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh \
    && bash miniconda.sh -b -p /root/miniconda \
    && /root/miniconda/bin/conda config --set always_yes yes --set changeps1 no \
    && /root/miniconda/bin/conda update -q conda \
    && /root/miniconda/bin/conda install -y \
        conda-build \
        anaconda-client \
    && /root/miniconda/bin/conda clean -tipsy \
    # config ssh service
    && mkdir /var/run/sshd \
    && echo 'root:singa' | chpasswd \
    # for ubuntu 16.04 prohibit
    && sed -i 's/PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    # SSH login fix. Otherwise user is kicked off after login
    && sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd \
    # dump environment variables into files, so that ssh can see also
    && env | grep _ >> /etc/environment

# Add conda to PATH. Doing this here so other RUN steps can be grouped above
ENV PATH /root/miniconda/bin:${PATH}

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
