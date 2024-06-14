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
# Docker Images


## Available tags

* `devel`, with SINGA and the development packages installed on Ubuntu16.04 (no GPU)
* `devel-cuda`, with SINGA, CUDA8.0, CUDNN5, and other development packages installed on Ubuntu16.04

## Use the existing Docker images

Users can pull the Docker images from Dockerhub via

    docker pull apache/singa:1.2.0-cpu-devel-ubuntu18.04
    # or
    docker pull apache/singa:1.2.0-cuda10.0-cudnn7.4.2-devel-ubuntu18.04

    For more versions see [here](https://hub.docker.com/r/apache/singa/tags)

Run the docker container using

    docker run -it apache/singa:1.2.0-cpu-devel-ubuntu18.04 /bin/bash
    # or
    nvidia-docker run -it apache/singa:1.2.0-cuda10.0-cudnn7.4.2-devel-ubuntu18.04 /bin/bash

The latest SINGA code is under the `singa` folder.

***Warning*** The code will be under root/singa for 1.2.0-cpu-devel-ubuntu18.04.

## Create new Docker images from Dockerfile

New Docker images could be created by executing the following command within the
Dockerfile folder, e.g., tool/docker/devel/

    docker build -t apache/singa:<TAG> -f Dockerfile

The `<TAG>` is named as

    VERSION-devel|runtime[-CUDA|CPU][-CUDNN]

* VERSION: e.g., 3.0.0
* devel: development images with all dependent libs' header files installed and SINGA's source code; runtime: the minimal images which can run SINGA programs.
* CUDA: cuda10.0, cuda9.0
* CUDNN: cudnn7

Here are some example tags:

`devel-cuda9-cudnn7`, `devel-cuda9-cudnn7`, `devel-cuda10-cudnn7`, `devel-cpu`, `runtime-gpu` and `runtime-cpu`


Please follow the existing Dockefiles under tool/docker/ to create other Dockefiles.
The folder structure is like

    level1: devel|runtime
    level2: Dockerfile, OS
    level3: Dockerfile, CUDA|MKLDNN


For example, the path of the Dockerfile for `devel-cuda9-cudnn7` is `tool/docker/devel/ubuntu/cuda9/Dockerfile`.
