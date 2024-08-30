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

This note is written for the repo maintainer.

To create conda package and upload, you can follow the following steps:

## 1. Build the docker image

To have the build environment, use the dockerfile from https://github.com/apache/singa/blob/dev/tool/conda/docker/cuda10/Dockerfile

For example:

    docker build SINGA_DIR/singa/tool/conda/docker/cpu -t nusdbsystem/singa:conda-cpu

## 2. Build the container from the image and access the container

For example:

    docker run -it -p 2222:22 --runtime nvidia --name singa_conda_cpu --rm nusdbsystem/singa:conda-cpu bash

## 3. Build conda packages

There are three versions of conda package: (i) CPU, (ii) GPU, and (iii) DIST (distributed training)

To build the specific versions, follow the steps:

(i) cd to the folder tool/conda/singa, follow the instruction in https://github.com/apache/singa/blob/dev/tool/conda/singa/README.md

For example, for CPU version:

    cd tool/conda/singa
    conda config --add channels conda-forge
    conda config --add channels nusdbsystem
    conda-build .  --python 3.6
    anaconda -t $ANACONDA_UPLOAD_TOKEN upload -u nusdbsystem -l main /root/miniconda/conda-bld/linux-64/singa-3.1.0-cpu_py36.tar.bz2

The above will generate the SINGA package. The next step will be wrapping it to generate SINGA-CPU / SINGA-GPU / SINGA-DIST conda packages.

(ii) For different versions, cd to different folder:

For CPU version, cd to tool/conda/cpu
For GPU version, cd to tool/conda/gpu
For DIST version, cd to tool/conda/dist

(iii) Generate the SINGA-CPU / SINGA-GPU / SINGA-DIST anaconda package and upload.

For an example of SINGA-CPU version:

    cd tool/conda/cpu
    conda-build .  --python 3.6
    anaconda -t $ANACONDA_UPLOAD_TOKEN upload -u nusdbsystem -l main /root/miniconda/conda-bld/linux-64/singa-cpu-3.1.0-py36.tar.bz2

Since SINGA-CPU, SINGA-GPU and SINGA-DIST packages are wrapped from SINGA package, the steps from (i) to (iii) are necessary.
