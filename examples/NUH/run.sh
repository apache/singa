#!/bin/bash

if [ $# -ne 2 ];then
  echo "Usage: run.sh [start|stop] num_procs"
  exit
fi

script_path=`readlink -f $0`
script_dir=`dirname $script_path`
example_dir=`dirname $script_dir`
singa_dir=`dirname $example_dir`
exec_path=${singa_dir}/build/pm
host_path=$script_dir/hostfile
model_path=$script_dir/mlp.conf
cluster_path=$script_dir/cluster.conf
ssh_options="-oStrictHostKeyChecking=no \
-oUserKnownHostsFile=/dev/null"

hosts=(`cat $host_path |cut -d ' ' -f 1`)
params=(`cat $host_path | cut -d ' ' -f 2`)
if [ $1 == "start" ]
then
  rm -rf $singa_dir/log*
  for (( i=0; i<$2; i++ ))
  do
   	cmd="source ~/.bash_profile; touch $singa_dir/$i.lock;\
      $exec_path  --hostfile=$script_dir/hostfile --procs_id=$i\
      --model=${modelfile} --cluster=${clusterfile}"
    echo ${hosts[$i]} $ssh_options  $cmd
    ssh $ssh_options ${hosts[$i]} $cmd &
  done
elif [ $1 == "stop" ]
then
  for (( idx=0 ; idx<$2 ; idx++ ))
  do
    echo "ssh ${hosts[$idx]} \"kill pm\""
    ssh $ssh_options ${hosts[$idx]} "killall -q pm"
    sleep 1
  done
fi
