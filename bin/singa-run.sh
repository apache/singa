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

usage="Usage: singa-run.sh -workspace=YOUR_WORKSPACE [ --resume ]\n
       # workspace should contain job.conf\n
       # set --resume if want to recover a job\n
       ### NOTICE ###\n
       # if you are using model.conf + cluster.conf,\n
       # please see how to combine them to a job.conf:\n
       # http://singa.incubator.apache.org/quick-start.html"

# check arguments
while [ $# != 0 ]; do
  if [[ $1 == "-workspace="* ]]; then
    workspace=$1
  elif [ $1 == "--resume" ]; then
    resume=1
  else
    echo -e $usage
    exit 1
  fi
  shift
done
if [ -z $workspace ]; then
  echo -e $usage
  exit 1
fi

# get environment variables
. `dirname "${BASH_SOURCE-$0}"`/singa-env.sh
# get workspace path
workspace=`cd "${workspace:11}">/dev/null; pwd`
job_conf=$workspace/job.conf
if [ ! -f $job_conf ]; then
  echo job.conf not exists in $workspace
  exit 1
fi
cd $SINGA_HOME

# start zookeeper
if [ $SINGA_MANAGES_ZK = true ]; then
  $SINGA_BIN/zk-service.sh start || exit 1
fi

# generate host file
host_file=$workspace/job.hosts
python $SINGA_HOME/tool/gen_hosts.py -conf=$job_conf \
                                     -hosts=$SINGA_CONF/hostfile \
                                     -output=$host_file \
                                     || exit 1

# generate unique job id
./singatool create 1>$workspace/job.id || exit 1
job_id=`cat $workspace/job.id`
echo Generate job id to $workspace/job.id [job_id = $job_id]

# set command to run singa
singa_run="./singa -workspace=$workspace -job=$job_id"
if [ ! -z $resume ]; then
  singa_run="$singa_run --resume"
fi
singa_sshrun="cd $SINGA_HOME; $singa_run"

# ssh and start singa processes
ssh_options="-oStrictHostKeyChecking=no \
-oUserKnownHostsFile=/dev/null \
-oLogLevel=quiet"
hosts=`cat $host_file | cut -d ' ' -f 1`
for i in ${hosts[@]} ; do
  if [ $i = localhost ] ; then
    echo Executing : $singa_run
    $singa_run &
  else
    echo Executing @ $i : $singa_sshrun
    ssh $ssh_options $i $singa_sshrun &
  fi
done

# generate pid list for this job
sleep 2
./singatool view $job_id 1>$workspace/job.pids || exit
echo Generate pid list to $workspace/job.pids
wait
