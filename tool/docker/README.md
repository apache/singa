# SINGA Docker Images

## Availabe images

TO BE UPDATED.

| Tag | OS version | devel/runtime | Device|CUDA/CUDNN|
|:----|:-----------|:--------------|:------|:---------|
|runtime| Ubuntu16.04|runtime|CPU|-|
|runtime| Ubuntu16.04|runtime|CPU|-|
|runtime-cuda| Ubuntu16.04|runtime|GPU|CUDA8.0+CUDNN5|
|devel| Ubuntu16.04|devel|CPU|-|
|devel-cuda| Ubuntu16.04|devel|GPU|CUDA8.0+CUDNN5|

## Usage

    docker pull nusdbsystem/singa:<Tag>
    docker run -it nusdbsystem/singa:<Tag> /bin/bash

* For the *devel* images, the container has a `incubator-singa` folder in the root directory,
which has the latest SINGA code. The code has been compiled into `incubator-singa/build` directory and PySINGA has been installed.
* For the *runtime* images, the container has only installed the PySINGA.

## Tag naming style

    singa:devel|runtime[-OS][-CUDA|OPENCL][-CUDNN]

* devel: development images with all dependent libs' header files installed and SINGA's source code;
* runtime: the minimal images which can run SINGA programs.
* OS: ubuntu, ubuntu14.04, centos, centos6
* CUDA: cuda, cuda8.0, cuda7.0
* CUDNN: cudnn, cudnn5, cudnn4
* OPENCL: opencl, opencl1.2

By default, if the version is not included in the tag, the latest stable version is used.
The default OS is ubuntu. The version is the latest stable version (e.g., 16.04 for now).
For -cuda version, the **cudnn** is included by default. Their versions are also the latest stable version, i.e., cuda-8.0 and cudnn-5 for now.
