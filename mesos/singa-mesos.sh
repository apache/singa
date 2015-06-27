#!/bin/bash

if [ $# -ne 2 ]; then
	echo "singa-mesos [--master/slave] [--start/--stop]"
	exit 1
fi

source mesos.conf

MSG="MESOS_HOME, MESOS_MASTER_IP, MESOS_SLAVE_IP, ZK_HOME or LIBPROCESS_IP not set"

if [ -z $MESOS_HOME ] || [ -z $MESOS_MASTER_IP ] || [ -z $LIBPROCESS_IP ] || [ -z $ZK_HOME ]; then
	echo $MSG
	exit 1
fi

if [ $1 == "--slave" ] && [ -z $MESOS_SLAVE_IP ]; then
	echo $MSG
	exit 1
fi

MASTER_COMMAND="sudo -E $MESOS_HOME/build/bin/mesos-master.sh --ip=$MESOS_MASTER_IP --work_dir=$MESOS_HOME --log_dir=$MESOS_HOME/logs --quiet > /dev/null &"

SLAVE_COMMAND="sudo -E $MESOS_HOME/build/bin/mesos-slave.sh --master=$MESOS_MASTER_IP:5050 --ip=$MESOS_SLAVE_IP --hostname=$MESOS_SLAVE_IP --log_dir=$MESOS_HOME/logs --isolation=cgroups/cpu,cgroups/mem --quiet  > /dev/null &"


if [ $2 == "--stop" ]; then
	if [ $1 == "--master" ]; then
		$ZK_HOME/bin/zkServer.sh stop
		sudo -E killall -KILL -r .*mesos-master
	else
		sudo -E killall -KILL -r .*mesos-slave
	fi
	rm -rf $MESOS_HOME/logs/*
fi

if [ $2 == "--start" ]; then
	if [ $1 == "--master" ]; then
		#start zookeeper service
		$ZK_HOME/bin/zkServer.sh start
		eval $MASTER_COMMAND
	else
		eval $SLAVE_COMMAND
	fi

	exit 0
fi
