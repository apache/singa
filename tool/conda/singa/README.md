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
# Package SINGA using conda-build

[conda-build](https://conda.io/docs/user-guide/tasks/build-packages/index.html) is a packaging tool like apt-get, which is associated with [anaconda cloud](https://anaconda.org/) for package management for both python and cpp libraries.


## Environment variables

We export the CUDA version if SINGA is compiled with CUDA enabled. The cuDNN version is fixed by SINGA and cuDNN is installed from [anaconda cloud](https://anaconda.org/anaconda/cudnn).

    # for SINGA with GPU, e.g. cuda9.0-cudnn7.3.1
    export CUDA=9.0

Then, we export a flag DIST to indicate if SINGA is compiled with distributed training enabled.

    # to enable distributed training: DIST=ON, otherwise: DIST=OFF
    export DIST=OFF

We need to export both CUDA and DIST for GPU version. For CPU-only version, we do not export CUDA and DIST.

## Instruction

After exporting the environment variables, we need to add the necessary conda channels

    conda config --add channels conda-forge
    conda config --add channels nusdbsystem

Then, we can execute the following commands to compile SINGA and package it

    conda-build .  --python 3.6

You will see the package path from the screen output, e.g., `xx/yy/singa-1.2.0-cpu.tar.bz2` or `xx/yy/singa-1.2.0-cudnn7.3.1_cuda9.0.tar.bz2`.

To clean the cache

    conda clean -ay
