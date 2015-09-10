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
#!/bin/bash

if [ $# -ne 2 ];then
  echo "Usage: run.sh [start|stop] num_procs"
  exit
fi

netconf=conv.conf

script_path=`readlink -f $0`
script_dir=`dirname $script_path`
example_dir=`dirname $script_dir`
singa_dir=`dirname $example_dir`
exec_path=${singa_dir}/build/singa
host_path=$script_dir/hostfile
ssh_options="-oStrictHostKeyChecking=no \
-oUserKnownHostsFile=/dev/null \
-oLogLevel=quiet"

hosts=(`cat $host_path |cut -d ' ' -f 1`)
if [ $1 == "start" ]
then
  count=0
  for i in ${hosts[@]}
  do
    cmd="touch $singa_dir/$count.lock;\
      $exec_path \
      -procsID=$count \
      -hostfile=$host_path \
      -cluster_conf=$script_dir/cluster.conf \
      -model_conf=$script_dir/$netconf; rm -f $singa_dir/$count.lock"
    echo $cmd
    ssh $ssh_options $i $cmd &
    count=$(($count+1))
    if [ $count -eq $2 ]
    then
      exit
    fi
  done
elif [ $1 == "stop" ]
then
  for (( idx=$2-1 ; idx>=0 ; idx-- ))
  do
    echo "ssh ${hosts[$idx]} \"kill singa\""
    ssh $ssh_options ${hosts[$idx]} "killall -q singa"
    sleep 1
  done
fi


