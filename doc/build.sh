#!/bin/bash

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


SPHINXBUILD="sphinx-build"
BUILDDIR="_build"
LANG_ARR=(en zh)

if [ "$1"x = "clean"x ]; then
  rm -rf $BUILDDIR/*
  rm -rf en/docs/examples
  echo "clean up $BUILDDIR"
fi


if [ "$1"x = "html"x ]; then
  cp -rf ../examples en/docs/model_zoo
  cp README.md en/develop/contribute-docs.md
  for (( i=0; i<${#LANG_ARR[@]}; i++)) do
    echo "building language ${LANG_ARR[i]} ..."
    if [ ${LANG_ARR[i]} = "en" ]; then
      $SPHINXBUILD -b html -c . -d $BUILDDIR/doctree ${LANG_ARR[i]} $BUILDDIR/html
      $SPHINXBUILD -b html -c . -d $BUILDDIR/doctree ${LANG_ARR[i]} $BUILDDIR/html/${LANG_ARR[i]}
    else
      $SPHINXBUILD -b html -c ${LANG_ARR[i]} -d $BUILDDIR/doctree ${LANG_ARR[i]} $BUILDDIR/html/${LANG_ARR[i]}
    fi  
  done
  ( cat Doxyfile ; echo "OUTPUT_DIRECTORY=$BUILDDIR/html/doxygen" ) | doxygen - 
fi
