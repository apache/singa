#!/bin/bash

#if [ $# -ne 1 ];then
#  echo "Usage: run.sh node_id"
#  exit
#fi
#trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM EXIT

#trap 'kill $(jobs -p)' EXIT SIGTERM SIGINT SIGHUP

DIR=`pwd`
TOPOLOGY="--topology_config=$SINGA_HOME/examples/mnist/topology.conf"
HOSTFILE="--hostfile=$SINGA_HOME/examples/mnist/hostfile"
LOG=singa_log_node_$1

cd $DIR
cd ../

COMMAND="build/pm --node_id=$1 $TOPOLOGY $HOSTFILE --v=3 2> $LOG"
#COMMAND="echo $TOPOLOGY >$LOG"
echo $COMMAND
eval $COMMAND
