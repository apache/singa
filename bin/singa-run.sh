#!/usr/bin/env bash
#
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
#
# Run a Singa job
#

usage="Usage: \n \
  (single node): singa-run.sh -cluster=YOUR_CONF_FILE -model=YOUR_CONF_FILE \n \
  (distributed): singa-run.sh -conf=YOUR_CONF_DIR \ 
  (the directory should contain cluster.conf/model.conf/hostfile)"

#if [ $# -le 0 ] || [ $# -ge 3 ] ; then
#  echo -e $usage
#  exit 1
#fi

valid_args=false

if [ $# = 1 ] ; then
  if [[ $1 = "-conf="* ]] ; then
    valid_args=true
    conf_path=${1:6}
    host_path=$conf_path/hostfile
  fi
elif [ $# = 2 ] ; then
  if [[ $1 = "-cluster="* ]] && [[ $2 = "-model="*  ]] ; then
    valid_args=true
  elif [[ $2 = "-cluster="* ]] && [[ $1 = "-model="*  ]] ; then
    valid_args=true
  fi
fi

if [ $valid_args = false ] ; then
  echo -e $usage
  exit 1 
fi

# get singa-base
BIN=`dirname "${BASH_SOURCE-$0}"`
BIN=`cd "$BIN">/dev/null; pwd`
BASE=`cd "$BIN/..">/dev/null; pwd`

cd $BASE

# clenup singa data
if [ -z $host_path ] ; then
  $BIN/singa-stop.sh 
else
  $BIN/singa-stop.sh $host_path
fi

# start zookeeper
$BIN/zk-service.sh start 2>/dev/null

# wait for zk service to be up
sleep 3

# check mode
if [ $# = 2 ] ; then
  # start singa process
  cmd="./singa "$@
  echo starting singa ...
  echo executing : $cmd
  exec $cmd
elif [ $# = 1 ] ; then
  # ssh and start singa processes
  ssh_options="-oStrictHostKeyChecking=no \
  -oUserKnownHostsFile=/dev/null \
  -oLogLevel=quiet"
  hosts=(`cat $host_path |cut -d ' ' -f 1`)
  for i in ${hosts[@]} ; do
    cmd="cd $BASE; \
        ./singa \
        -cluster=$conf_path/cluster.conf \
        -model=$conf_path/model.conf"
    echo executing @ $i : $cmd
    ssh $ssh_options $i $cmd &
  done
  wait
fi

# cleanup singa data
if [ -z $host_path ] ; then
  $BIN/singa-stop.sh
else
  $BIN/singa-stop.sh $host_path
fi
