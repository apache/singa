#!/bin/bash

if [ $# -ne 2 ]; then
	echo "singa-mesos [--master/slave] [--start/--stop]"
	exit 1
fi

if [ -z $MESOS_HOME ]; then
	echo "MESOS_HOME not set"
	exit 1
fi

if [ -z $MESOS_MASTER_IP ]; then
	echo "MESOS_MASTER_IP not set"
	exit 1
fi

if [ $1 == "--slave" ]; then
	if [ -z $MESOS_SLAVE_IP ]; then
		echo "MESOS_SLAVE_IP not set"
		exit 1
	fi
	if [ -z $SINGA_HOME ]; then
		echo "SINGA_HOME not set"
		exit 1 
	fi
fi

MASTER_COMMAND="$MESOS_HOME/build/bin/mesos-master.sh --ip=$MESOS_MASTER_IP --work_dir=$MESOS_HOME --log_dir=$MESOS_HOME/logs --quiet > /dev/null &"

SLAVE_COMMAND="$MESOS_HOME/build/bin/mesos-slave.sh --master=$MESOS_MASTER_IP:5050 --ip=$MESOS_SLAVE_IP --hostname=$MESOS_SLAVE_IP --log_dir=$MESOS_HOME/logs --isolation=cgroups/cpu,cgroups/mem --quiet  > /dev/null &"


if [ $2 == "--stop" ]; then
	if [ $1 == "--master" ]; then
		killall -KILL lt-mesos-master
	else
		killall -KILL lt-mesos-slave
	fi
fi

if [ $2 == "--start" ]; then
	if [ $1 == "--master" ]; then
		eval $MASTER_COMMAND
	else
		echo $SLAVE_COMMAND
		eval $SLAVE_COMMAND
	fi

	exit 0
fi
