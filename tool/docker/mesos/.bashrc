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

export LIBRARY_PATH=/opt/OpenBLAS/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/OpenBLAS/lib:/usr/local/lib:$LD_LIBRARY_PATH
export CPLUS_INCLUDE_PATH=/opt/OpenBLAS/include:$CPLUS_INCLUDE_PATH
export PATH=/opt/jdk1.8.0_60/bin:/opt/bin:$PATH
export HADOOP_HOME=/opt/hadoop-2.6.0
export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin
export JAVA_HOME=/opt/jdk1.8.0_60
alias ls="ls --color=always"
# some more ls aliases
alias ll="ls -alF"
alias la="ls -A"
alias l="ls -CF"
export SINGA_HOME=/root/incubator-singa
export PATH=$PATH:$SINGA_HOME/bin
