# Docker Images


## Available tags

* `devel`, with SINGA and the development packages installed on Ubuntu16.04 (no GPU)
* `devel-cuda`, with SINGA, CUDA8.0, CUDNN5, and other development packages installed on Ubuntu16.04

## Use the existing Docker images

Users can pull the Docker images from Dockerhub via

    docker pull apache/singa:devel
    # or
    docker pull apache/singa:devel-cuda

Run the docker container using

    docker run -it apache/singa:devel /bin/bash
    # or
    docker run -it apache/singa:devel-cuda /bin/bash

The latest SINGA code is under the `incubator-singa` folder.

## Create new Docker images from Dockerfile

New Docker images could be created by executing the following command within the
Dockerfile folder, e.g., tool/docker/devel/

    docker build -t singa:<TAG> -f Dockerfile

The `<TAG>` is named as

    devel|runtime[-CUDA|CPU][-CUDNN]

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
