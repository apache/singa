---
id: version-2.0.0-build
title: Build SINGA from Source
original_id: build
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

The source files could be downloaded either as a [tar.gz file](https://dist.apache.org/repos/dist/dev/singa/), or as a git repo

```shell
$ git clone https://github.com/apache/singa.git
$ cd singa/
```

If you want to contribute code to SINGA, refer to [contribute-code page](contribute-code.md) for the steps and requirements.

## Use Conda to build SINGA

Conda-build is a building tool that installs the dependent libraries from anaconda cloud and executes the building scripts.

To install conda-build (after installing conda)

```shell
conda install conda-build
```

### Build CPU Version

To build the CPU version of SINGA

```shell
conda build tool/conda/singa/
```

The above commands have been tested on Ubuntu (14.04, 16.04 and 18.04) and macOS 10.11. Refer to the [Travis-CI page](https://travis-ci.org/apache/singa) for more information.

### Build GPU Version

To build the GPU version of SINGA, the building machine must have Nvida GPU, and the CUDA driver (>= 384.81), CUDA toolkit (>=9) and cuDNN (>=7) must have be installed. The following two Docker images provide the building environment:

1. apache/singa:conda-cuda9.0
2. apache/singa:conda-cuda10.0

Once the building environment is ready, you need to export the CUDA version first, and then run conda command to build SINGA

```shell
export CUDA=x.y (e.g. 9.0)
conda build tool/conda/singa/
```

### Post Processing

The location of the generated package file (`.tar.gz`) is shown on the screen. The generated package can be installed directly,

```shell
conda install -c conda-forge --use-local <path to the package file>
```

or uploaded to anaconda cloud for others to download and install. You need to register an account on anaconda for [uploading the package](https://docs.anaconda.com/anaconda-cloud/user-guide/getting-started/).

```shell
conda install anaconda-client
anaconda login
anaconda upload -l main <path to the package file>
```

After uploading the package to the cloud, you can see it on [Anaconda Cloud](https://anaconda.org/) website or via the following command

```shell
conda search -c <anaconda username> singa
```

Each specific SINGA package is identified by the version and build string. To install a specific SINGA package, you need to provide all the information, e.g.,

```shell
conda install -c <anaconda username> -c conda-forge singa=2.1.0.dev=cpu_py36
```

To make the installation command simple, you can create the following additional packages which depend on the latest CPU and GPU SINGA packages.

```console
# for singa-cpu
conda build tool/conda/cpu/  --python=3.6
conda build tool/conda/cpu/  --python=3.7
# for singa-gpu
conda build tool/conda/gpu/  --python=3.6
conda build tool/conda/gpu/  --python=3.7
```

Therefore, when you run

```shell
conda install -c <anaconda username> -c conda-forge singa-xpu
```

(`xpu` is either 'cpu' or 'gpu'), the corresponding real SINGA package is installed as the dependent library.

## Use native tools to build SINGA on Ubuntu

Refer to SINGA [Dockerfiles](https://github.com/apache/singa/blob/master/tool/docker/devel/ubuntu/cuda9/Dockerfile#L30) for the instructions of installing the dependent libraries on Ubuntu 16.04. You can also create a Docker container using the [devel images]() and build SINGA inside the container. To build SINGA with GPU, MKLDNN, Python and unit tests, run the following instructions

```shell
mkdir build    # at the root of singa folder
cd build
cmake -DENABLE_TEST=ON -DUSE_CUDA=ON -DUSE_MKLDNN=ON -DUSE_PYTHON3=ON ..
make
cd python
pip install .
```

The details of the CMake options are explained in the last section of this page. The last command install the Python package. You can also run `pip install -e .`, which creates symlinks instead of copying the Python files into the site-package folder.

If SINGA is compiled with ENABLE_TEST=ON, you can run the unit tests by

```shell
$ ./bin/test_singa
```

You can see all the testing cases with testing results. If SINGA passes all tests, then you have successfully installed SINGA.

## Use native tools to Build SINGA on Centos7

Building from source will be different for Centos7 as package names differ.Follow the instructions given below.

### Installing dependencies

Basic packages/libraries

```shell
sudo yum install freetype-devel libXft-devel ncurses-devel openblas-devel blas-devel lapack devel atlas-devel kernel-headers unzip wget pkgconfig zip zlib-devel libcurl-devel cmake curl unzip dh-autoreconf git python-devel glog-devel protobuf-devel
```

For build-essential

```shell
sudo yum group install "Development Tools"
```

For installing swig

```shell
sudo yum install pcre-devel
wget http://prdownloads.sourceforge.net/swig/swig-3.0.10.tar.gz
tar xvzf swig-3.0.10.tar.gz
cd swig-3.0.10.tar.gz
./configure --prefix=${RUN}
make
make install
```

For installing gfortran

```shell
sudo yum install centos-release-scl-rh
sudo yum --enablerepo=centos-sclo-rh-testing install devtoolset-7-gcc-gfortran
```

For installing pip and other packages

```shell
sudo yum install epel-release
sudo yum install python-pip
pip install matplotlib numpy pandas scikit-learn pydot
```

### Installation

Follow steps 1-5 of _Use native tools to build SINGA on Ubuntu_

### Testing

You can run the unit tests by,

```shell
$ ./bin/test_singa
```

You can see all the testing cases with testing results. If SINGA passes all tests, then you have successfully installed SINGA.

## Compile SINGA on Windows

Instructions for building on Windows with Python support can be found [install-win page](install-win.md).

## More details about the compilation options

### USE_MODULES (deprecated)

If protobuf and openblas are not installed, you can compile SINGA together with them

```shell
$ In SINGA ROOT folder
$ mkdir build
$ cd build
$ cmake -DUSE_MODULES=ON ..
$ make
```

cmake would download OpenBlas and Protobuf (2.6.1) and compile them together with SINGA.

You can use `ccmake ..` to configure the compilation options. If some dependent libraries are not in the system default paths, you need to export the following environment variables

```shell
export CMAKE_INCLUDE_PATH=<path to the header file folder>
export CMAKE_LIBRARY_PATH=<path to the lib file folder>
```

### USE_PYTHON

Option for compiling the Python wrapper for SINGA,

```shell
$ cmake -DUSE_PYTHON=ON ..
$ make
$ cd python
$ pip install .
```

### USE_CUDA

Users are encouraged to install the CUDA and [cuDNN](https://developer.nvidia.com/cudnn) for running SINGA on GPUs to get better performance.

SINGA has been tested over CUDA 9/10, and cuDNN 7. If cuDNN is installed into non-system folder, e.g. /home/bob/local/cudnn/, the following commands should be executed for cmake and the runtime to find it

```shell
$ export CMAKE_INCLUDE_PATH=/home/bob/local/cudnn/include:$CMAKE_INCLUDE_PATH
$ export CMAKE_LIBRARY_PATH=/home/bob/local/cudnn/lib64:$CMAKE_LIBRARY_PATH
$ export LD_LIBRARY_PATH=/home/bob/local/cudnn/lib64:$LD_LIBRARY_PATH
```

The cmake options for CUDA and cuDNN should be switched on

```shell
# Dependent libs are install already
$ cmake -DUSE_CUDA=ON ..
$ make
```

### USE_MKLDNN

User can enable MKL-DNN to enhance the performance of CPU computation.

Installation guide of MKL-DNN could be found [here](https://github.com/intel/mkl-dnn#installation).

SINGA has been tested over MKL-DNN v0.17.2.

To build SINGA with MKL-DNN support:

```shell
# Dependent libs are installed already
$ cmake -DUSE_MKLDNN=ON ..
$ make
```

### USE_OPENCL

SINGA uses opencl-headers and viennacl (version 1.7.1 or newer) for OpenCL support, which can be installed using via

```shell
# On Ubuntu 16.04
$ sudo apt-get install opencl-headers, libviennacl-dev
# On Fedora
$ sudo yum install opencl-headers, viennacl
```

Additionally, you will need the OpenCL Installable Client Driver (ICD) for the platforms that you want to run OpenCL on.

- For AMD and nVidia GPUs, the driver package should also install the correct OpenCL ICD.
- For Intel CPUs and/or GPUs, get the driver from the [Intel website.](https://software.intel.com/en-us/articles/opencl-drivers) Note that the drivers provided on that website only supports recent CPUs and Iris GPUs.
- For older Intel CPUs, you can use the `beignet-opencl-icd` package.

Note that running OpenCL on CPUs is not currently recommended because it is slow. Memory transfer is on the order of whole seconds (1000's of ms on CPUs as compared to 1's of ms on GPUs).

More information on setting up a working OpenCL environment may be found [here](https://wiki.tiker.net/OpenCLHowTo).

If the package version of ViennaCL is not at least 1.7.1, you will need to build it from source:

Clone [the repository from here](https://github.com/viennacl/viennacl-dev), checkout the `release-1.7.1` tag and build it. Remember to add its directory to `PATH` and the built libraries to `LD_LIBRARY_PATH`.

To build SINGA with OpenCL support (tested on SINGA 1.1):

```shell
$ cmake -DUSE_OPENCL=ON ..
$ make
```

### PACKAGE

This setting is used to build the Debian package. Set PACKAGE=ON and build the package with make command like this:

```shell
$ cmake -DPACKAGE=ON
$ make package
```

## FAQ

- Q: Error from 'import singa'

  A: Please check the detailed error from `python -c "from singa import _singa_wrap"`. Sometimes it is caused by the dependent libraries, e.g. there are multiple versions of protobuf, missing of cudnn, numpy version mismatch. Following steps show the solutions for different cases

  1. Check the cudnn and cuda. If cudnn is missing or not match with the wheel version, you can download the correct version of cudnn into ~/local/cudnn/ and

     ```shell
     $ echo "export LD_LIBRARY_PATH=/home/<yourname>/local/cudnn/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
     ```

  2. If it is the problem related to protobuf. You can install protobuf (3.6.1) from source into a local folder, say ~/local/; Decompress the tar file, and then

     ```shell
     $ ./configure --prefix=/home/<yourname>local
     $ make && make install
     $ echo "export LD_LIBRARY_PATH=/home/<yourname>/local/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
     $ source ~/.bashrc
     ```

  3. If it cannot find other libs including python, then create virtual env using `pip` or `conda`;

  4. If it is not caused by the above reasons, go to the folder of `_singa_wrap.so`,

     ```shell
     $ python
     >> import importlib
     >> importlib.import_module('_singa_wrap')
     ```

     Check the error message. For example, if the numpy version mismatches, the error message would be,

     ```shell
     RuntimeError: module compiled against API version 0xb but this version of numpy is 0xa
     ```

     Then you need to upgrade the numpy.

* Q: Error from running `cmake ..`, which cannot find the dependent libraries.

  A: If you haven't installed the libraries, install them. If you installed the libraries in a folder that is outside of the system folder, e.g. /usr/local, you need to export the following variables

  ```shell
  $ export CMAKE_INCLUDE_PATH=<path to your header file folder>
  $ export CMAKE_LIBRARY_PATH=<path to your lib file folder>
  ```

- Q: Error from `make`, e.g. the linking phase

  A: If your libraries are in other folders than system default paths, you need to export the following varaibles

  ```shell
  $ export LIBRARY_PATH=<path to your lib file folder>
  $ export LD_LIBRARY_PATH=<path to your lib file folder>
  ```

* Q: Error from header files, e.g. 'cblas.h no such file or directory exists'

  A: You need to include the folder of the cblas.h into CPLUS_INCLUDE_PATH, e.g.,

  ```shell
  $ export CPLUS_INCLUDE_PATH=/opt/OpenBLAS/include:$CPLUS_INCLUDE_PATH
  ```

* Q:While compiling SINGA, I get error `SSE2 instruction set not enabled`

  A:You can try following command:

  ```shell
  $ make CFLAGS='-msse2' CXXFLAGS='-msse2'
  ```

* Q:I get `ImportError: cannot import name enum_type_wrapper` from google.protobuf.internal when I try to import .py files.

  A: You need to install the python binding of protobuf, which could be installed via

  ```shell
  $ sudo apt-get install protobuf
  ```

  or from source

  ```shell
  $ cd /PROTOBUF/SOURCE/FOLDER
  $ cd python
  $ python setup.py build
  $ python setup.py install
  ```

* Q: When I build OpenBLAS from source, I am told that I need a Fortran compiler.

  A: You can compile OpenBLAS by

  ```shell
  $ make ONLY_CBLAS=1
  ```

  or install it using

  ```shell
  $ sudo apt-get install libopenblas-dev
  ```

* Q: When I build protocol buffer, it reports that `GLIBC++_3.4.20` not found in `/usr/lib64/libstdc++.so.6`?

  A: This means the linker found libstdc++.so.6 but that library belongs to an older version of GCC than was used to compile and link the program. The program depends on code defined in the newer libstdc++ that belongs to the newer version of GCC, so the linker must be told how to find the newer libstdc++ shared library. The simplest way to fix this is to find the correct libstdc++ and export it to LD_LIBRARY_PATH. For example, if GLIBC++\_3.4.20 is listed in the output of the following command,

        $ strings /usr/local/lib64/libstdc++.so.6|grep GLIBC++

  then you just set your environment variable as

        $ export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

* Q: When I build glog, it reports that "src/logging_unittest.cc:83:20: error: ‘gflags’ is not a namespace-name"

  A: It maybe that you have installed gflags with a different namespace such as "google". so glog can't find 'gflags' namespace. Because it is not necessary to have gflags to build glog. So you can change the configure.ac file to ignore gflags.

        1. cd to glog src directory
        2. change line 125 of configure.ac  to "AC_CHECK_LIB(gflags, main, ac_cv_have_libgflags=0, ac_cv_have_libgflags=0)"
        3. autoreconf

  After this, you can build glog again.

* Q: When using virtual environment, every time I run pip install, it would reinstall numpy. However, the numpy would not be used when I `import numpy`

  A: It could be caused by the `PYTHONPATH` which should be set to empty when you are using virtual environment to avoid the conflicts with the path of the virtual environment.

* Q: When compiling PySINGA from source, there is a compilation error due to the missing of <numpy/objectarray.h>

  A: Please install numpy and export the path of numpy header files as

        $ export CPLUS_INCLUDE_PATH=`python -c "import numpy; print numpy.get_include()"`:$CPLUS_INCLUDE_PATH

* Q: When I run SINGA in Mac OS X, I got the error "Fatal Python error: PyThreadState_Get: no current thread Abort trap: 6"

  A: This error happens typically when you have multiple version of Python on your system and you installed SINGA via pip (this problem is resolved for installation via conda), e.g, the one comes with the OS and the one installed by Homebrew. The Python linked by PySINGA must be the same as the Python interpreter. You can check your interpreter by `which python` and check the Python linked by PySINGA via `otool -L <path to _singa_wrap.so>`. To fix this error, compile SINGA with the correct version of Python. In particular, if you build PySINGA from source, you need to specify the paths when invoking [cmake](http://stackoverflow.com/questions/15291500/i-have-2-versions-of-python-installed-but-cmake-is-using-older-version-how-do)

        $ cmake -DPYTHON_LIBRARY=`python-config --prefix`/lib/libpython2.7.dylib -DPYTHON_INCLUDE_DIR=`python-config --prefix`/include/python2.7/ ..

  If installed PySINGA from binary packages, e.g. debian or wheel, then you need to change the python interpreter, e.g., reset the \$PATH to put the correct path of Python at the front position.
