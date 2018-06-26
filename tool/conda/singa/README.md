<!--
    Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.
-->
# Package Singa using conda-build

[conda-build](https://conda.io/docs/user-guide/tasks/build-packages/index.html) is a packaging tool like apt-get, which is associated with [anaconda cloud](https://anaconda.org/) for package management for both python and cpp libraries.


## Environment variables

Build string is a part of the conda package specification. We include the cuda and cudnn version in it if Singa is compiled with CUDA enabled.

	# for singa with gpu, e.g. cuda8.0-cudnn7.0.5
    export BUILD_STR=cudax.y-cudnna.b.c

    # for singa running only on cpu
    export BUILD_STR=cpu


To package Singa with CUDA and CUDNN,

    export CUDNN_PATH=<path to cudnn folder>

this folder should include a subfolder `include/cudnn.h` for the header file, and another subfolder `lib64` for the shared libraries. The BUILD_STR and CUDNN_PATH must be consistent. For example, if CUDNN_PATH is set, then BUILD_STR must be like cudax.y-cudnna.b.c. CUDNN must be provided if we want to compiled Singa with CUDA enabled.

## Instruction

After exporting the environment variables, execute the following command to compile Singa and package it

    conda-build .  --python 3.6  (or 2.7)

You will see the package path from the screen output.

To clean the cache

    conda clean -ay
