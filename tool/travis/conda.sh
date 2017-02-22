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

# to build SINGA package and upload it to anaconda

set -x

# anaconda login user name
USER=nusdbsystem
OS=$TRAVIS_OS_NAME-64

export PATH="$HOME/miniconda/bin:$PATH"
conda config --set anaconda_upload no

# save the package at given folder, then we can upload using singa-*.tar.bz2
suffix=`TZ=Asia/Singapore date +%Y-%m-%d-%H-%M-%S`
export CONDA_BLD_PATH=~/conda-bld-$suffix
mkdir $CONDA_BLD_PATH

conda build tool/conda/

# turn off debug to hide the token in travis log
set +x
anaconda -t $ANACONDA_UPLOAD_TOKEN upload -u $USER -l nightly $CONDA_BLD_PATH/$OS/singa-*.tar.bz2 --force
