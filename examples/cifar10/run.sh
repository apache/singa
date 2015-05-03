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


