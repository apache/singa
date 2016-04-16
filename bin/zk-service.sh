#!/usr/bin/env bash
#
#/**
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
# manage ZooKeeper service
#

usage="Usage: zk-service.sh [start|stop]"

if [ $# != 1 ]; then
  echo $usage
  exit 1
fi

# get environment variables
. `dirname "${BASH_SOURCE-$0}"`/singa-env.sh

# check if singa manages zookeeper service
if [ $SINGA_MANAGES_ZK != true ]; then
  echo "Singa does not manage a valid zookeeper service (SINGA_MANAGES_ZK != true)"
  exit 1
fi

# check zookeeper installation
if [ ! -d $ZK_HOME ]; then
  echo "zookeeper not found at $ZK_HOME"
  echo "if you do not have zookeeper service, please install:"
  echo "    $SINGA_HOME/thirdparty/install.sh zookeeper"
  echo "otherwise, please set ZK_HOME correctly"
  exit 1
fi

# get command
case $1 in
  start)
    # start zk service
    # check zoo.cfg
    if [ ! -f $ZK_HOME/conf/zoo.cfg ]; then
      echo "zoo.cfg not found, create from sample.cfg"
      cp $ZK_HOME/conf/zoo_sample.cfg $ZK_HOME/conf/zoo.cfg
    fi
    # cd to SINGA_HOME as zookeeper.out will be here
    cd $SINGA_HOME
    $ZK_HOME/bin/zkServer.sh start
    ;;

  stop)
    # stop zk service
    $ZK_HOME/bin/zkServer.sh stop
    ;;

  *)
    echo $usage
    exit 1
esac

