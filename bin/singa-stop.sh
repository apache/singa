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
# Clean up singa processes and zookeeper metadata
#

usage="Usage: \n \
      (local process): singa-stop.sh \n \
      (distributed): singa-stop.sh HOST_FILE"

if [ $# -gt 1 ]; then
  echo -e $usage
  exit 1
fi

BIN=`dirname "${BASH_SOURCE-$0}"`
BIN=`cd "$BIN">/dev/null; pwd`
BASE=`cd "$BIN/..">/dev/null; pwd`
ZKDATA_DIR="/tmp/zookeeper"

PROC_NAME="*singa"
HOST_FILE=$1


# kill singa processes
if [ $# = 0 ] ; then
  echo kill singa @ localhost ...
  cmd="killall -s SIGKILL "$PROC_NAME
  $cmd
elif [ $# = 1 ] ; then
  ssh_options="-oStrictHostKeyChecking=no \
  -oUserKnownHostsFile=/dev/null \
  -oLogLevel=quiet"
  hosts=(`cat $HOST_FILE |cut -d ' ' -f 1`)
  for i in ${hosts[@]} ; do
    cmd="killall -s SIGKILL -r "$PROC_NAME
    echo kill singa @ $i ...
    if [ $i == localhost ] ; then
      $cmd
    else
      ssh $ssh_options $i $cmd
    fi
  done
fi

# close zookeeper
. $BIN/zk-service.sh stop 2>/dev/null

echo cleanning metadata in zookeeper ...
# remove zk data
rm -r $ZKDATA_DIR

