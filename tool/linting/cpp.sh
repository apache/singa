#!/usr/bin/env bash
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

export PATH="$HOME/miniconda/bin:$PATH"

# cpplint
find src/api/ \
    src/core/ \
    src/proto/ \
    src/utils/ \
    include/singa/core/ \
    include/singa/utils/ \
    src/model/operation/ \
    include/singa/io/communicator.h \
    src/io/communicator.cc \
    test/singa/ -iname *.h -o -iname *.cc | xargs cpplint --quiet --verbose=5

CPPLINTRESULT=$?

if [ $CPPLINTRESULT -ne 0 ]; then
  echo $CPPLINTRESULT
  echo "cpplint not passed"
  exit 1
else
  echo "cpplint passed"
fi

exit 0
