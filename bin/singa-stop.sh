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
# clean up singa processes and zookeeper metadata
#

# get environment variables
. `dirname "${BASH_SOURCE-$0}"`/singa-env.sh
cd $SINGA_HOME

# kill singa processes
host_file=$SINGA_CONF/hostfile
ssh_options="-oStrictHostKeyChecking=no \
             -oUserKnownHostsFile=/dev/null \
             -oLogLevel=quiet"
hosts=`cat $host_file | cut -d ' ' -f 1`
singa_kill="killall -s SIGKILL -r singa"
for i in ${hosts[@]}; do
  echo Kill singa @ $i ...
  if [ $i == localhost ]; then
    $singa_kill
  else
    ssh $ssh_options $i $singa_kill
  fi
done
# wait for killall command
sleep 2

# remove zk data
echo Cleanning metadata in zookeeper ...
./singatool cleanup || exit 1
