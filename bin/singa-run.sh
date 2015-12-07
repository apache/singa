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
# run a singa job
#

usage="Usage: singa-run.sh -conf <job config file> [ other arguments ]\n
        -resume                 : if want to recover a job\n
        -exec <path to mysinga> : if want to use own singa driver\n
       ### NOTICE ###\n
        if you are using model.conf + cluster.conf,\n
        please see how to combine them to a job.conf:\n
        http://singa.incubator.apache.org/quick-start.html"

# parse arguments
# make sure we have '-conf' and remove '-exec'
exe=./singa
while [ $# != 0 ]; do
  if [ $1 == "-exec" ]; then
    shift
    exe=$1
  elif [ $1 == "-conf" ]; then
    shift
    conf=$1
  else
    args="$args $1"
  fi
  shift
done
if [ -z $conf ]; then
  echo -e $usage
  exit 1
fi

# get environment variables
. `dirname "${BASH_SOURCE-$0}"`/singa-env.sh

# change conf to an absolute path
conf_dir=`dirname "$conf"`
conf_dir=`cd "$conf_dir">/dev/null; pwd`
conf_base=`basename "$conf"`
job_conf=$conf_dir/$conf_base
if [ ! -f $job_conf ]; then
  echo $job_conf not exists
  exit 1
fi
cd $SINGA_HOME

# generate unique job id
job_id=`./singatool create`
[ $? == 0 ] || exit 1
echo Unique JOB_ID is $job_id

# generate job info dir
# format: job-JOB_ID-YYYYMMDD-HHMMSS
log_dir=$SINGA_LOG/job-info/job-$job_id-$(date '+%Y%m%d-%H%M%S')
mkdir -p $log_dir
echo Record job information to $log_dir

# generate host file
host_file=$log_dir/job.hosts
./singatool genhost $job_conf 1>$host_file || exit 1

# set command to run singa
singa_run="$exe $args -conf $job_conf \
            -singa_conf $SINGA_HOME/conf/singa.conf \
            -singa_job $job_id" 
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
./singatool view $job_id 1>$log_dir/job.pids || exit 1
wait
