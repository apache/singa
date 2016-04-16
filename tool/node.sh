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
#!/bin/bash
if [[ $# < 2 || ! -f $2 ]]
then
  echo "Usage: process/folder management"
  echo "[cat, create, delete, kill, ls, ps, reset, scp, ssh] hostfile [args]"
  echo "   cat hostfile file--- cat the file on every node in hostfile"
  echo "   create hostfile folder--- create the folder on every node in hostfile"
  echo "   delete hostfile folder--- delete the folder on every node in hostfile"
  echo "   kill hostfile job_name---  kill the job on every node in hostfile"
  echo "   ls hostfile folder--- list the folder on every node in hostfile"
  echo "   ps hostfile job_name---  ps aux|grep job_name on every node in hostfile"
  echo "   reset hostfile folder--- delete and create the folder on every node in hostfile"
  echo "   scp hostfile local_dir [remote_dir]--- copy the local_dir to remote_dir on every node in hostfile, if remote_dir is omitted, remote_dir=local_dir"
  echo "   ssh hostfile--- test whether the nodes in hostfile are alive"
  echo "each line in hostfile is a node name followed by a space and other fields"
  exit
fi

ssh_options="-oStrictHostKeyChecking=no \
-oUserKnownHostsFile=/dev/null \
-oLogLevel=quiet"

hosts=(`cat $2 |cut -d ' ' -f 1`)

for i in ${hosts[@]}
do
  if [ $1 == "cat" ]
  then
    cmd="cat $3"
  elif [ $1 == "create" -o $1 == "reset" ]
  then
    cmd="mkdir -p $3"
  elif [ $1 == "delete" -o $1 == "reset" ]
  then
    cmd="rm -rf $3"
  elif [ $1 == "kill" ]
  then
    cmd="ps ax|pgrep $3 |xargs kill"
  elif [ $1 == "ls" ]
  then
    cmd="ls -l $3"
  elif [ $1 == "scp" ]
  then
    local_dir=$3
    remote_dir=$3
    if [ $# -eq 4 ]
    then
      remote_dir=$4
    fi
    r=''
    if [[ -d $3 ]]
    then
      r='-r'
    fi
    echo "scp $r $local_dir $i:$remote_dir"
    scp $r $local_dir $i:$remote_dir
  elif [ $1 == "ssh" ]
  then
    cmd="exit"
  elif [ $1 == "ps" ]
  then
    cmd="ps ax|pgrep $3"
  else
    echo "Incorrect commands:" $1
  fi
  if [ $1 != "scp" ]
  then
    echo $cmd
    ssh $i $cmd
  fi
done
