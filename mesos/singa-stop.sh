#!/bin/bash

source mesos.conf
source singa.conf

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
for i in `cat framework_log | grep frameworkId | awk '{print \$8}'`
do
	echo "Killing framework $i"
	COMMAND="curl -d \"frameworkId=$i\" -X POST http://$MESOS_MASTER_IP:5050/master/shutdown"
	eval $COMMAND
	echo
done

killall -KILL singa_scheduler
