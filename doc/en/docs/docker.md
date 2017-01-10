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

    devel|runtime[-OS][-CUDA|OPENCL][-CUDNN]

* devel: development images with all dependent libs' header files installed and SINGA's source code; runtime: the minimal images which can run SINGA programs.
* OS: ubuntu, ubuntu14.04, centos, centos6
* CUDA: cuda, cuda8.0, cuda7.0
* CUDNN: cudnn, cudnn5, cudnn4
* OPENCL: opencl, opencl1.2

By default, if the version is not included in the tag, the latest stable version is used.
The default OS is Ubuntu. The version is the latest stable version (e.g., 16.04 for now).
For -cuda version, the **cudnn** is included by default. Their versions are also the latest stable version, i.e., cuda8.0 and cudnn5 for now.

Here are some example tags,

`devel`, `devel-cuda`, `runtime`, `runtime-cuda`, `devel-centos7-cuda`, `devel-ubuntu14.04`, `devel-ubuntu14.04-cuda7.5-cudnn4`

Please follow the existing Dockefiles under tool/docker/ to create other Dockefiles.
The folder structure is like

    level1: devel|runtime
    level2: Dockerfile, OS
    level3: Dockerfile, CUDA|OPENCL
    level4: CUDNN

For example, the path of the Dockerfile for `devel-cuda` is `tool/docker/devel/cuda/Dockerfile`.
