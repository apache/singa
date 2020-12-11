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

# TODO(wangwei) install other libs and test. It has bugs now.

# Change tags to build with different cuda/cudnn versions:
FROM nvidia/cuda:7.5-cudnn5-devel-centos6

# install dependencies
RUN yum -y update \
    && yum install -y \
    git \
    wget \
    openssh-server \
    cmake \
    && yum clean all \
    # download singa source
    # RUN git clone https://github.com/apache/singa.git
    # config ssh service
    && mkdir /var/run/sshd \
    && ssh-keygen -t rsa -f /etc/ssh/ssh_host_rsa_key \
    && ssh-keygen -t dsa -f /etc/ssh/ssh_host_dsa_key \
    && sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config \
    && sed -ri 's/#UsePAM no/UsePAM no/g' /etc/ssh/sshd_config \
    && echo 'root:singa' | chpasswd \
    #RUN sed -i 's/PermitRootLogin without-password/PermitRootLogin yes/' /etc/ssh/sshd_config
    # SSH login fix. Otherwise user is kicked off after login
    #RUN sed 's@session\s*required\s*pam_loginuid.so@session optional pam_loginuid.so@g' -i /etc/pam.d/sshd
    # dump environment variables into files, so that ssh can see also
    && env | grep _ >> /etc/environment

EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
