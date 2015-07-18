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
# run a Singa job
#

usage="Usage: singa-run.sh -conf=CONF_DIR 
      (CONF_DIR should contain cluster.conf && model.conf)"
# usage="Usage: \n
#       (single process): singa-run.sh -cluster=YOUR_CONF_FILE -model=YOUR_CONF_FILE \n
#       (multi-process): singa-run.sh -conf=YOUR_CONF_DIR 
#       (the directory should contain cluster.conf && model.conf)"

# check arguments
if [ $# != 1 ] || [[ $1 != "-conf="* ]]; then
  echo $usage
  exit 1
fi

# get environment variables
. `dirname "${BASH_SOURCE-$0}"`/singa-env.sh
# get workspace path
workspace=`cd "${1:6}">/dev/null; pwd`

# start zookeeper
if [ $SINGA_MANAGES_ZK = true ]; then
  $SINGA_BIN/zk-service.sh start || exit 1
fi

# cleanup old processes and data
$SINGA_BIN/singa-stop.sh || exit 1

# generate host file
host_file=$workspace/job.hosts
python $SINGA_HOME/tool/gen_hosts.py -conf=$workspace/cluster.conf \
                                     -hosts=$SINGA_CONF/hostfile \
                                     -output=$host_file \
                                     || exit 1

# ssh and start singa processes
ssh_options="-oStrictHostKeyChecking=no \
-oUserKnownHostsFile=/dev/null \
-oLogLevel=quiet"
hosts=`cat $host_file |cut -d ' ' -f 1`
# cd to SINGA_HOME as it need conf/singa.conf
cd $SINGA_HOME
singa_run="./singa -cluster=$workspace/cluster.conf -model=$workspace/model.conf"
singa_sshrun="cd $SINGA_HOME; ./singa -cluster=$workspace/cluster.conf \
              -model=$workspace/model.conf"
for i in ${hosts[@]} ; do
  if [ $i = localhost ] ; then
    echo executing : $singa_run
    $singa_run &
  else
    echo executing @ $i : $singa_sshrun
    ssh $ssh_options $i $singa_sshrun &
  fi
done
wait
