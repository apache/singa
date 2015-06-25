#!/bin/bash

if [ -z $MESOS_HOME ]; then
	echo "MESOS_HOME not set"
	exit 1
fi

if [ -z $MESOS_MASTER_IP ]; then
	echo "MESOS_MASTER_IP not set"
	exit 1
fi

if [ -z $SINGA_HOME ]; then
	echo "SINGA_HOME not set"
	exit 1
fi

# read from the file
COMMAND="cat framework_log | grep frameworkId | awk '{print \$7}'"
#COMMAND="./singa_scheduler $MESOS_MASTER_IP:5050 $1 $SINGA_HOME > framework_log 2>&1 &"
ID=`eval $COMMAND`
COMMAND="curl -d \"frameworkId=$ID\" -X POST http://$MESOS_MASTER_IP:5050/master/shutdown"
eval $COMMAND
killall -KILL singa_scheduler
