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
FROM nvidia/cuda:10.2-runtime-centos7

MAINTAINER singa dev@singa.apache.org

# install dependencies
RUN yum install -y epel-release python-pip 

RUN pip install singa -f http://singa.apache.org/docs/next/wheel-gpu.html --trusted-host singa.apache.org

# config ssh service
RUN mkdir /var/run/sshd \
    && echo 'root:singa' | chpasswd \
    && sed -ri 's/^#?PermitRootLogin\s+.*/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -ri 's/UsePAM yes/#UsePAM yes/g' /etc/ssh/sshd_config \
    && mkdir /root/.ssh


EXPOSE 22

CMD ["/usr/sbin/sshd", "-D"]
