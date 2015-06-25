#!/bin/bash

if [ $# -ne 1 ]; then
	echo "singa-submit <ncpus>"
fi

if [ -z $MESOS_HOME ]; then
	echo "MESOS_HOME not set"
	exit 1
fi

if [ -z $MESOS_MASTER_IP ]; then
	echo "MESOS_MASTER_IP not set"
	exit 1
fi

if [ -z $LIBPROCESS_IP ]; then
	echo "LIBPROCESS_IP not set"
	exit 1 
fi

if [ -z $SINGA_HOME ]; then
	echo "SINGA_HOME not set"
	exit 1
fi

COMMAND="./singa_scheduler $MESOS_MASTER_IP:5050 $1 $SINGA_HOME > framework_log 2>&1 &"
eval $COMMAND
