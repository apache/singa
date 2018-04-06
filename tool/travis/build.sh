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


if [[ "$TRAVIS_SECURE_ENV_VARS" == "false" ]];
then
  if [[ "$TRAVIS_OS_NAME" == "osx" ]];
  then
    export CMAKE_LIBRARY_PATH=/usr/local/opt/openblas/lib:/usr/local/opt/protobuf/lib:$CMAKE_LIBRARY_PATH;
    export CMAKE_INCLUDE_PATH=/usr/local/opt/openblas/include:/usr/local/opt/protobuf/include:$CMAKE_INCLUDE_PATH;
    mkdir build && cd build;
    cmake -DUSE_CUDA=OFF -DUSE_PYTHON=OFF -DENABLE_TEST=ON -DProtobuf_PROTOC_EXECUTABLE=/usr/local/opt/protobuf/bin/protoc ..;
  else
    mkdir build && cd build;
    cmake -DUSE_CUDA=OFF -DUSE_PYTHON=OFF -DENABLE_TEST=ON -DUSE_MODULES=ON ..
  fi
  make;
  ./bin/test_singa --gtest_output=xml:./../gtest.xml;
else
  bash -e tool/travis/conda.sh;
fi
