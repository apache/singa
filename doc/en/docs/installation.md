# Installation

## Dependencies

### Required
* google protobuf (>=2.5,<3)
* blas (tested with openblas >=0.2.10)
* cmake (>=2.6)


### Optional
* glog
* opencv (tested with 2.4.8)
* lmdb (tested with 0.9)
* cuda (tested with 6.5, 7.0 and 7.5)
* cudnn (v4 and v5)

PySINGA has additional dependencies

* python(==2.7)
* pip(>=1.5)
* swig(>=3.0)
* numpy(>=1.11.0)
* openblas (>=0.2.10)

Users are encouraged to install the cuda and [cudnn](https://developer.nvidia.com/cudnn) for running SINGA on GPUs to
get better performance.
Most of the dependent libraries could be installed via package mangers like
apt-get or homebrew.

    # for ubuntu users, tested on 14.04
    sudo apt-get install libprotobuf-dev libopenblas-dev libopencv-dev protobuf-compiler libgoogle-glog-dev liblmdb-dev python2.7-dev python-pip python-numpy

    # for Mac OS users
    brew install -vd glog lmdb
    brew tap homebrew/science
    brew install opencv
    brew install openblas
    brew tap homebrew/python
    brew install python
    brew install numpy  --with-openblas


## Install PySINGA

### From wheel

After installing the dependencies for SINGA and PySINGA, please download the correct binary:

    # Ubuntu/Linux 64-bit, CPU only, Python 2.7, Protobuf 2.5
    $ export SINGA_WHEEL_URL=http://comp.nus.edu.sg/~dbsystem/singa/assets/file/pb2.5/singa-1.0.0-cp27-none-linux_x86_64.whl

    # Ubuntu/Linux 64-bit, CPU only, Python 2.7, Protobuf 2.6
    $ export SINGA_WHEEL_URL=http://comp.nus.edu.sg/~dbsystem/singa/assets/file/pb2.6/singa-1.0.0-cp27-none-linux_x86_64.whl

    # Ubuntu/Linux 64-bit, GPU enabled, Python 2.7, Protobuf 2.5, CUDA toolkit 7.5 and CuDNN v5
    $ export SINGA_WHEEL_URL=http://comp.nus.edu.sg/~dbsystem/singa/assets/file/pb2.5-cuda7.5-cudnn5/singa-1.0.0-cp27-none-linux_x86_64.whl

    # Ubuntu/Linux 64-bit, GPU enabled, Python 2.7, Protobuf 2.6, CUDA toolkit 7.5 and CuDNN v5
    $ export SINGA_WHEEL_URL=http://comp.nus.edu.sg/~dbsystem/singa/assets/file/pb2.6-cuda7.5-cudnn5/singa-1.0.0-cp27-none-linux_x86_64.whl

Then, run the following command

    $ sudo pip install --upgrade $SINGA_WHEEL_URL

If you do not have sudo right, you can run `pip install` in a python virtual environment.
Note that in python virtual environment, you may need to reset the `PYTHONPATH` to empty
to avoid the conflicts of system path and virtual environment path.


### From source

Please compile SINGA from source (see the next section) with the 'USE_PYTHON' option on,
and then run the following commands,

    # under the build directory
    $ cd python
    $ sudo pip install .

If you are using a virtual environment, you can ignore the `sudo` keyword.

Developers can build the wheel file via

    # under the build directory
    $ cd python
    $ python setup.py bdist_wheel

The generated wheel file is under "dist" directory.
To build cnmem into the wheel file, please change CMakeLists.txt by replacing
'SHARED' with 'STATIC'.


## Build SINGA from source

### From the downloaded `tar.gz` file:

Extract the downloaded. If using CUDA, CNMeM needs to be fetched:
   
    $ cd $SINGA_ROOT/lib/cnmem/
    $ git clone https://github.com/NVIDIA/cnmem

### From Git:

Please clone the newest code from [Github](https://github.com/apache/incubator-singa) and execute the following commands,

    $ git clone https://github.com/apache/incubator-singa.git
    $ cd incubator-singa/

If you use CUDA, then [CNMeM](https://github.com/NVIDIA/cnmem) is necessary,
which could be downloaded as

    $ git submodule init
    $ git submodule update


### Linux & MacOS

GCC (>=4.8.1) is required to compile SINGA on Linux.
For Mac OS users, you can use either GCC or Clang.

In SINGA_ROOT, execute the following commands for compiling SINGA,

    $ mkdir build && cd build
    $ cmake ..
    $ make
    $ make install

Note that if you are using CUDNN and it is not installed under system default
folder, you need to let cmake know the paths to CUDNN,

    $ export CMAKE_INCLUDE_PATH=<path to cudnn>/include:$CMAKE_INCLUDE_PATH
    $ export CMAKE_LIBRARY_PATH=<path to cudnn>/lib64:$CMAKE_LIBRARY_PATH

You can use `ccmake ..` to configure the compilation options including using
generating python binding and changing the installation folder.
If the dependent libraries are not in the system default paths, you need to export
the following environment variables

    export CMAKE_INCLUDE_PATH=<path to your header file folder>
    export CMAKE_LIBRARY_PATH=<path to your lib file folder>

After compiling SINGA, you can run the unit tests by

    $ ./bin/test_singa

You can see all the testing cases with testing results. If SINGA passes all
tests, then you have successfully installed SINGA. Please proceed to try the examples!


### Windows
To be added.


## FAQ

* Q: Error from running `cmake ..`, which cannot find the dependent libraries.

    A: If you haven't installed the libraries, please install them. If you installed
    the libraries in a folder that is outside of the system folder, e.g. /usr/local,
    please export the following variables

        export CMAKE_INCLUDE_PATH=<path to your header file folder>
        export CMAKE_LIBRARY_PATH=<path to your lib file folder>


* Q: Error from `make`, e.g. the linking phase

    A: If your libraries are in other folders than system default paths, you need
    to export the following varaibles

    $ export LIBRARY_PATH=<path to your lib file folder>
    $ export LD_LIBRARY_PATH=<path to your lib file folder>


* Q: Error from header files, e.g. 'cblas.h no such file or directory exists'

    A: You need to include the folder of the cblas.h into CPLUS_INCLUDE_PATH,
    e.g.,

        $ export CPLUS_INCLUDE_PATH=/opt/OpenBLAS/include:$CPLUS_INCLUDE_PATH

* Q:While compiling SINGA, I get error `SSE2 instruction set not enabled`

    A:You can try following command:

        $ make CFLAGS='-msse2' CXXFLAGS='-msse2'

* Q:I get `ImportError: cannot import name enum_type_wrapper` from google.protobuf.internal when I try to import .py files.

    A: You need to install the python binding of protobuf, which could be installed via

        $ sudo apt-get install protobuf

    or from source

        $ cd /PROTOBUF/SOURCE/FOLDER
        $ cd python
        $ python setup.py build
        $ python setup.py install

* Q: When I build OpenBLAS from source, I am told that I need a Fortran compiler.

    A: You can compile OpenBLAS by

        $ make ONLY_CBLAS=1

    or install it using

        $ sudo apt-get install libopenblas-dev

* Q: When I build protocol buffer, it reports that GLIBC++_3.4.20 not found in /usr/lib64/libstdc++.so.6.

    A9: This means the linker found libstdc++.so.6 but that library
    belongs to an older version of GCC than was used to compile and link the
    program. The program depends on code defined in
    the newer libstdc++ that belongs to the newer version of GCC, so the linker
    must be told how to find the newer libstdc++ shared library.
    The simplest way to fix this is to find the correct libstdc++ and export it to
    LD_LIBRARY_PATH. For example, if GLIBC++_3.4.20 is listed in the output of the
    following command,

        $ strings /usr/local/lib64/libstdc++.so.6|grep GLIBC++

    then you just set your environment variable as

        $ export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH

* Q: When I build glog, it reports that "src/logging_unittest.cc:83:20: error: ‘gflags’ is not a namespace-name"

    A: It maybe that you have installed gflags with a different namespace such as "google". so glog can't find 'gflags' namespace.
    Because it is not necessary to have gflags to build glog. So you can change the configure.ac file to ignore gflags.

        1. cd to glog src directory
        2. change line 125 of configure.ac  to "AC_CHECK_LIB(gflags, main, ac_cv_have_libgflags=0, ac_cv_have_libgflags=0)"
        3. autoreconf

    After this, you can build glog again.

* Q: When using virtual environment, everytime I run pip install, it would reinstall numpy. However, the numpy would not be used when I `import numpy`

    A: It could be caused by the `PYTHONPATH` which should be set to empty when you are using virtual environment to avoid the conflicts with the path of
    the virtual environment.
