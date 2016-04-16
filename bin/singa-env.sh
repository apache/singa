#!/usr/bin/env bash
#
#/**
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
# set singa environment variables, includes:
#   * SINGA_HOME
#   * SINGA_BIN
#   * SINGA_CONF
#   * SINGA_LOG
#   * ZK_HOME
#   * SINGA_MANAGES_ZK
#

# exit if varaiables already set
[ -z $SINGA_ENV_DONE ] || exit 0

# set SINGA_BIN
if [ -z $SINGA_BIN ]; then
  SINGA_BIN=`dirname "${BASH_SOURCE-$0}"`
  SINGA_BIN=`cd "$SINGA_BIN">/dev/null; pwd`
fi

# set SINGA_HOME
if [ -z $SINGA_HOME ]; then
  SINGA_HOME=`cd "$SINGA_BIN/..">/dev/null; pwd`
fi

# set SINGA_CONF
if [ -z $SINGA_CONF ]; then
  SINGA_CONF=$SINGA_HOME/conf
fi

# set SINGA_LOG
if [ -z $SINGA_LOG ]; then
  # add -confdir arg, so no need to run under SINGA_HOME
  SINGA_LOG=`"$SINGA_HOME"/singatool getlogdir -confdir "$SINGA_CONF"`
  [ $? == 0 ] || exit 1
fi

# set ZK_HOME
if [ -z $ZK_HOME ]; then
  ZK_HOME=$SINGA_HOME/thirdparty/zookeeper-3.4.6
  SINGA_MANAGES_ZK=true
fi

# set SINGA_MANAGES_ZK
if [ -z $SINGA_MANAGES_ZK ]; then
  SINGA_MANAGES_ZK=false
fi

# mark that we have done all
SINGA_ENV_DONE=1
