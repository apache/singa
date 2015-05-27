#!/usr/bin/env bash
#
#/**
# * Copyright 2015 The Apache Software Foundation
# *
# * Licensed to the Apache Software Foundation (ASF) under one
# * or more contributor license agreements.  See the NOTICE file
# * distributed with this work for additional information
# * regarding copyright ownership.  The ASF licenses this file
# * to you under the Apache License, Version 2.0 (the
# * "License"); you may not use this file except in compliance
# * with the License.  You may obtain a copy of the License at
# *
# *     http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.
# */
# 
# Run a Singa job
#

usage="Usage: singa-run.sh"

#if [ $# -le 0 ]; then
#  echo $usage
#  exit 1
#fi
  
BIN=`dirname "${BASH_SOURCE-$0}"`
BIN=`cd "$BIN">/dev/null; pwd`
BASE=`cd "$BIN/..">/dev/null; pwd`

#get argument
cmd=$1

cd $BASE

$BIN/zk-service.sh start

#wait for zk service to be up
sleep 3

echo starting singa ...

#echo $@
./singa $@

echo stopping singa ...

$BIN/zk-service.sh stop
