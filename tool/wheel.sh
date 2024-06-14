#!/bin/bash
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

# this script should be launched at the root of the singa source folder
# it build the cpu-only and cuda enabled wheel packages for py3.6, 3.7 and 3.8

rm -rf dist

# build cpu only wheel packages
rm -rf build
/opt/python/cp39-cp39/bin/python setup.py bdist_wheel
rm -rf build
/opt/python/cp310-cp310/bin/python setup.py bdist_wheel
rm -rf build
/opt/python/cp311-cp311/bin/python setup.py bdist_wheel

# build cuda enabled wheel packages
export SINGA_CUDA=ON
rm -rf build
/opt/python/cp39-cp39/bin/python setup.py bdist_wheel
rm -rf build
/opt/python/cp310-cp310/bin/python setup.py bdist_wheel
rm -rf build
/opt/python/cp311-cp311/bin/python setup.py bdist_wheel

# repair the wheel files in dist/*.whl and store the results into wheelhouse/
/opt/python/cp311-cp311/bin/python setup.py audit
