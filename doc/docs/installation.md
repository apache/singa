# Building SINGA from source

## Dependencies

### Required
* Google Protobuf (>=2.5,<3)
* BLAS (tested with OpenBLAS >=0.2.10)
* CUDA (tested with 6.5, 7.0 and 7.5)
* CUDNN (v4 and v5)
* cmake (>=2.6)

Users must install the above mandatory libraries.
Currently CUDA and CUDNN are also mandatory, but it would become optional later.

### Optional
* Glog
* OpenCV (tested with 2.4.8)
* LMDB (tested with 0.9)


## Instructions

Please clone the newest code from [Github](https://github.com/apache/incubator-singa) and execute the following commands,


    $ git clone https://github.com/apache/incubator-singa.git
    $ cd incubator-singa/
    # switch to dev branch
    $ git checkout dev


If you use CUDA, then [CNMeM](https://github.com/NVIDIA/cnmem) is necessary,
which could be downloaded as

    $ git submodule init
    $ git submodule update


### Linux OS

GCC (>=4.8.1) is required to compile SINGA on Linux OS.
In SINGA_ROOT, execute the following commands for compiling SINGA,

    $ mkdir build && cd build
    # generate Makefile for compilation
    $ cmake ..
    # compile SINGA
    $ make

Note that if you are using CUDNN, you need to let cmake know the paths to CUDNN,

    $ export CMAKE_INCLUDE_PATH=<path to cudnn>/include:$CMAKE_INCLUDE_PATH
    $ export CMAKE_LIBRARY_PATH=<path to cudnn>/lib64:$CMAKE_LIBRARY_PATH

You can use `cmake ..` to configure the compilation options including using
LMDB, GLOG, etc.

After compiling SINGA, you can run the unit tests by

    $ ./bin/test_singa

You can see all the testing cases with testing results. If SINGA passes all
tests, then you have successfully installed SINGA. Please proceed to try the examples!


### MacOS

Currently only Linux OS is officially support.

### Windows

Currently only Linux OS is officially support.


# Install SINGA Python Module

SINGA provide a python binding for python programers. Users can either install from source or 
from pre-built wheel file.

## Install from source

### Required
* python(==2.7)   
* pip(>=1.5)
* SWIG(>=3.0)   
* numpy(>=1.11.0)   
* Google protobuf(>=2.5,<3)   


### Configuration
To build SINGA python package, users should turn on python building switch in cmake config file: "CMakeList.txt"

    OPTION(USE_PYTHON "Generate py wrappers" ON)

### Instructions
Follow the instructions in the above sections to build SINGA from source,

After that, execute the following commands:

    # under the build directory
    $ cd python
    $ sudo pip install . 

Then singa package should be installed in the corresponding python library. 

## Pip Install from wheel 

Install pip if it is not already installed:

    $ sudo apt-get install python-pip python-dev

Then, select the correct binary to install:

    # Ubuntu/Linux 64-bit, CPU only, Python 2.7, Protobuf 2.5
    $ export SINGA_WHEEL_URL=http://comp.nus.edu.sg/~dbsystem/singa/assets/file/pb2.5/singa-1.0.0-cp27-none-linux_x86_64.whl

    # Ubuntu/Linux 64-bit, CPU only, Python 2.7, Protobuf 2.6
    $ export SINGA_WHEEL_URL=http://comp.nus.edu.sg/~dbsystem/singa/assets/file/pb2.6/singa-1.0.0-cp27-none-linux_x86_64.whl

    # Ubuntu/Linux 64-bit, GPU enabled, Python 2.7, Protobuf 2.5, CUDA toolkit 7.5 and CuDNN v5
    $ export SINGA_WHEEL_URL=http://comp.nus.edu.sg/~dbsystem/singa/assets/file/pb2.5-cuda7.5-cudnn5/singa-1.0.0-cp27-none-linux_x86_64.whl
   
    # Ubuntu/Linux 64-bit, GPU enabled, Python 2.7, Protobuf 2.6, CUDA toolkit 7.5 and CuDNN v5
    $ export SINGA_WHEEL_URL=http://comp.nus.edu.sg/~dbsystem/singa/assets/file/pb2.6-cuda7.5-cudnn5/singa-1.0.0-cp27-none-linux_x86_64.whl
   
Install SINGA:

    $ sudo pip install --upgrade $SINGA_WHEEL_URL

### build wheel file from source

Users can build wheel file from source. After build SINGA, execute the following commands:

    # under the build directory
    $ cd python
    $ python setup.py bdist_wheel

Then users may get built wheel file under "dist" directory
