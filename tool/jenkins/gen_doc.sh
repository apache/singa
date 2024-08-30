#!/usr/bin/env bash
#/**
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


# This script is used by Jenkins to update Singa website.
# Run this script in runtime docker container.

echo Install PySinga, generate HTML files and update Singa website

conda update singa
COMMIT=`git rev-parse --short HEAD`
cd doc
# generate the html files
bash build.sh html
# checkout the current website files
svn co https://svn.apache.org/repos/asf/incubator/singa/site/trunk
# overwrite the existing files
cp -r _build/html/* trunk/
# track newly added files and commit
cd trunk
svn add --force * --auto-props --parents --depth infinity -q
svn commit -m "update the docs by jenkins for commit $COMMIT"
