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

# if no env var (i.e., token), then do normal build and test;
# otherwise use conda to build and package
if [[ "$TRAVIS_SECURE_ENV_VARS" == "false" ]];
then
  if [[ "$TRAVIS_OS_NAME" == "linux" ]];
  then
    echo "nothing to install"
#    sudo apt-get -qq update;
#    sudo apt-get -qq -y install libopenblas-dev libprotobuf-dev protobuf-compiler;
  else
    brew update;
    # brew tap homebrew/science;
    brew install openblas protobuf;
  fi
else
  # install miniconda
  if [[ "$TRAVIS_OS_NAME" == "linux" ]];
  then
    wget https://repo.continuum.io/miniconda/Miniconda2-latest-Linux-x86_64.sh -O miniconda.sh;
  else
    wget https://repo.continuum.io/miniconda/Miniconda2-latest-MacOSX-x86_64.sh -O miniconda.sh;
  fi
  bash miniconda.sh -b -p $HOME/miniconda
  export PATH="$HOME/miniconda/bin:$PATH"
  hash -r
  conda config --set always_yes yes --set changeps1 no
  conda update -q conda
  conda install conda-build
  conda install anaconda-client
  conda config --add channels conda-forge
fi
