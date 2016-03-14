#!/bin/bash

#/**
# * Copyright 2015 The Apache Software Foundation
# *
# * Licensed to the Apache Software Foundation (ASF) under one
# * or more contributor license agreements.  See the NOTICE file
# * distributed with this work for additional information
# * regarding copyright ownership.  The ASF licenses this file
# * to you under the Apache License, Version 2.0 (the
# * "License"); you may not use this file except in compliance
# * with the License.  You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */

source /root/.bashrc
#download
cd /opt
wget -c http://archive.apache.org/dist/mesos/0.22.0/mesos-0.22.0.tar.gz
wget https://www.comp.nus.edu.sg/~dinhtta/files/mesos_patch 
tar -zxvf mesos-0.22.0.tar.gz

#patch and install mesos
cd /opt/mesos-0.22.0
patch -p5 < ../mesos_patch
mkdir build; cd build
../configure
make
sudo make install

