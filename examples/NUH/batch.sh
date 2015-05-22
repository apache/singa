#!/bin/bash

for nservers in 1
do
  for nthreads in 2 4
  do
    for nworkers in 1 2 4 8 16
    do
      echo "nworkers: $nworkers" >examples/mnist/cluster.conf
      echo "nservers: $nservers" >>examples/mnist/cluster.conf
      echo "nthreads_per_server: $nthreads" >>examples/mnist/cluster.conf
      echo  "workspace:\" /data1/wangwei/singa\"">>examples/mnist/cluster.conf
      cat examples/mnist/cluster.conf
      nprocs=$(($nworkers+$nservers))
      log=log1k/${nworkers}w${nservers}s${nthreads}t
      echo  $log $nprocs
      ./examples/mnist/run.sh start $nprocs >$log 2>&1
      sleep 4

      while true
      do
        nstopped=0
        to=$(($nprocs-1))
        for worker in $(eval echo "{0..$to}")
        do
          if [ ! -e /home/wangwei/program/singa/$worker.lock ]
          then
            echo "$worker.lock is free"
            nstopped=$(($nstopped+1))
          fi
        done
        if [ $nstopped -eq $(($nprocs)) ]
        then
          break
        else
          sleep 5
        fi
      done
    done
  done
done
