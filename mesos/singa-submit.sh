#!/bin/bash

source singa.conf
source mesos.conf

MSG="SINGA_HOME, SINGA_WORKDIR, LIBPROCESS_IP or MESOS_MASTER_IP is not set."

if [ -z $MESOS_MASTER_IP ] || [ -z $SINGA_HOME ] || [ -z $SINGA_WORKDIR ] || [ -z $LIBPROCESS_IP ]; then
	echo $MSG
	exit 1
fi

COMMAND="./singa_scheduler $MESOS_MASTER_IP:5050 --singa_home=$SINGA_HOME --singa_workdir=$SINGA_WORKDIR $@ > framework_log 2>&1 &"
eval $COMMAND
