# Installation

## Dependencies

### Required
* google protobuf (>=2.5,<3)
* blas (tested with openblas >=0.2.10)
* cmake (>=2.6)


### Optional
* cuda (tested with 6.5, 7.0 and 7.5)
* cudnn (v4 and v5)
* opencv (tested with 2.4.8)
* lmdb (tested with 0.9)
* glog
* opencl-headers and viennacl (version 1.7.1 or newer) for OpenCL support

PySINGA (the Python binding) has additional dependencies

* python(==2.7)
* pip(>=1.5)
* swig(>=3.0.10)
* numpy(>=1.11.0)

Users are encouraged to install the cuda and [cudnn](https://developer.nvidia.com/cudnn) for running SINGA on GPUs to
get better performance.
Most of the dependent libraries could be installed from source or via package mangers like
apt-get, homebrew, pip and anaconda. Please refer to FAQ for problems caused by the path setting of the dependent libraries.


### apt-get
The following instructions are tested on Ubuntu 14.04 for installing dependent libraries.

    # required libraries
    $ sudo apt-get install libprotobuf-dev libopenblas-dev protobuf-compiler

    # optional libraries
    $ sudo apt-get install python2.7-dev python-pip python-numpy
    $ sudo apt-get install libopencv-dev libgoogle-glog-dev liblmdb-dev

Please note that PySINGA requires swig >=3.0, which could be installed via
apt-get on Ubuntu 16.04; but it has to be installed from source for other Ubuntu versions including 14.04.

### homebrew
The following instructions are tested on Mac OS X Yosemite (10.10.5) for installing dependent libraries.

    # required libraries
    $ brew tap homebrew/science
    $ brew install openblas
    $ brew install protobuf260

    # optional libraries
    $ brew tap homebrew/python
    $ brew install python
    $ brew install opencv
    $ brew install -vd glog lmdb

By default, openblas is installed into /usr/local/opt/openblas. To let the compiler (and cmake) know the openblas
path, please export

    $ export CMAKE_INCLUDE_PATH=/usr/local/opt/openblas/include:$CMAKE_INCLUDE_PATH
    $ export CMAKE_LIBRARY_PATH=/usr/local/opt/openblas/lib:$CMAKE_LIBRARY_PATH

To let the runtime know the openblas path, please export

    $ export LD_LIBRARY_PATH=/usr/local/opt/openblas/library:$LD_LIBRARY_PATH

### pip and anaconda for PySINGA
pip and anaconda could be used to install python packages, e.g. numpy.
Python virtual environment is recommended to run PySINGA.
To use pip with virtual environment,

    # install virtualenv
    $ pip install virtualenv
    $ virtualenv pysinga
    $ source pysinga/bin/activate
    $ pip install numpy

To use anaconda with virtual environment,

    $ conda create --name pysinga python=2
    $ source activate pysinga
    $ conda install numpy

After installing numpy, please export the header path of numpy.i as

    $ export CPLUS_INCLUDE_PATH=`python -c "import numpy; print numpy.get_include()"`:$CPLUS_INCLUDE_PATH


## Install PySINGA

### From wheel

After installing the dependencies for SINGA and PySINGA, please download the correct binary:

    # Ubuntu/Linux 64-bit, CPU only, Python 2.7, Protobuf 2.5
    $ export SINGA_WHEEL_URL=http://comp.nus.edu.sg/~dbsystem/singa/assets/file/pb2.5/singa-1.0.0-cp27-none-linux_x86_64.whl

    # Ubuntu/Linux 64-bit, CPU only, Python 2.7, Protobuf 2.6
    $ export SINGA_WHEEL_URL=http://comp.nus.edu.sg/~dbsystem/singa/assets/file/pb2.6/singa-1.0.0-cp27-none-linux_x86_64.whl

    # Mac OSX (10.11), CPU only, Python 2.7, Protobuf 2.5
    $ export SINGA_WHEEL_URL=http://comp.nus.edu.sg/~dbsystem/singa/assets/file/pb2.5/singa-1.0.0-cp27-none-macosx_10_11_intel.whl

    # Mac OSX (10.11), CPU only, Python 2.7, Protobuf 2.6
    $ export SINGA_WHEEL_URL=http://comp.nus.edu.sg/~dbsystem/singa/assets/file/pb2.6/singa-1.0.0-cp27-none-macosx_10_11_intel.whl

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

Please compile SINGA from source (see the next section) with the 'USE_PYTHON' option on (`cmake -DUSE_PYTHON=ON ..`),
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

The source files could be downloaded either as a [tar.gz file](https://dist.apache.org/repos/dist/dev/incubator/singa/1.0.0/apache-singa-incubating-1.0.0-RC2.tar.gz), or as a git repo

    $ git clone https://github.com/apache/incubator-singa.git
    $ cd incubator-singa/

    # If you use CUDA, then CNMeM is necessary
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
generating python binding and changing the installation folder. Alternatively,
the options could be switched on by

    $ cmake -DUSE_CUDA=ON -DUSE_PYTHON=ON ..

If the dependent libraries are not in the system default paths, you need to export
the following environment variables

    export CMAKE_INCLUDE_PATH=<path to your header file folder>
    export CMAKE_LIBRARY_PATH=<path to your lib file folder>

After compiling SINGA, you can run the unit tests by

    $ ./bin/test_singa

You can see all the testing cases with testing results. If SINGA passes all
tests, then you have successfully installed SINGA. Please proceed to try the examples!

### OpenCL support (Linux)

Install `opencl-headers` and `libviennacl-dev` (Ubuntu 16) or `viennacl` (Fedora).

Additionally, you will need the OpenCL Installable Client Driver (ICD) for the platforms that you want to run OpenCL on.

* For AMD and nVidia GPUs, the driver package should also install the correct OpenCL ICD.
* For Intel CPUs and/or GPUs, get the driver from the [Intel website.](https://software.intel.com/en-us/articles/opencl-drivers) Note that the drivers provided on that website only supports recent CPUs and Iris GPUs.
* For older Intel CPUs, you can use the `beignet-opencl-icd` package.

Note that running OpenCL on CPUs is not currently recommended because it is slow. Memory transfer is on the order of whole seconds (1000's of ms on CPUs as compared to 1's of ms on GPUs).

More information on setting up a working OpenCL environment may be found [here](https://wiki.tiker.net/OpenCLHowTo).

If the package version of ViennaCL is not at least 1.7.1, you will need to build it from source:

Clone [the repository from here](https://github.com/viennacl/viennacl-dev), checkout the `release-1.7.1` tag and build it.
Remember to add its directory to `PATH` and the built libraries to `LD_LIBRARY_PATH`.

To build SINGA with OpenCL support, you need to pass the flag during cmake:

    cmake .. -DUSE_OPENCL=ON

### Windows

For the dependent library installation, please refer to [Dependencies](dependencies.md).
After all the dependencies are successfully installed, just run the following commands to
generate the VS solution in cmd under singa folder:

    $ md build && cd build
    $ cmake -G "Visual Studio 14" -DUSE_CUDA=OFF -DUSE_PYTHON=OFF ..

The default project generated by the command is 32-bit version. You can also
specify a 64-bit version project by:

    $ md build && cd build
    $ cmake -G "Visual Studio 14 Win64" -DUSE_CUDA=OFF -DUSE_PYTHON=OFF ..

If you get error outputs like "Could NOT find xxxxx" indicating a dependent
library missing, please configure your library file and include path for cmake or the system.
For example, you get an error "Could NOT find CBLAS" and suppose you installed
openblas header files at "d:\include" and openblas library at "d:\lib". You should run the
following command to specify your cblas parameters in cmake:

    $ cmake -G "Visual Studio 14" -DUSE_CUDA=OFF -DUSE_PYTHON=OFF -DCBLAS_INCLUDE_DIR="d:\include" -DCBLAS_LIBRARIES="d:\lib\libopenblas.lib" -DProtobuf_INCLUDE_DIR=<include dir of protobuf> -DProtobuf_LIBRARIES=<path to libprotobuf.lib> -DProtobuf_PROTOC_EXECUTABLE=<path to protoc.exe> -DGLOG_INCLUDE_DIR=<include dir of glog> -DGLOG_LIBRARIES=<path to libglog.lib> ..
    

To find out the parameters you need to specify for some special libraries, you
can run the following command:

    $ cmake -LAH

If you use cmake GUI tool in windows, please make sure you configure the right
parameters for the singa solution by select "Advanced" box. After generating the VS project,
please open the "singa.sln" project file under
the "build" folder and compile it as a normal VS solution. You will find the
unit tests file named "test_singa" in the project binary folder.
If you get errors when running test_singa.exe due to libglog.dll/libopenblas.dll missing,
please just copy the dll files into the same folder as test_singa.exe

## FAQ

* Q: Error from 'import singa' using PySINGA installed from wheel.

    A: Please check the detailed error from `python -c  "from singa import _singa_wrap"`. Sometimes it is
    caused by the dependent libraries, e.g. there are multiple versions of protobuf or missing of cudnn. Following
    steps show the solutions for different cases
    1. check the cudnn and cuda and gcc versions, cudnn5 and cuda7.5 and gcc4.8/4.9 are preferred. if gcc is 5.0, then downgrade it.
       if cudnn is missing or not match with the wheel version, you can download the correct version of cudnn into ~/local/cudnn/ and
        ```
        echo "export LD_LIBRARY_PATH=/home/<yourname>/local/cudnn/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc
        ```
    2. if it is the problem related to protobuf, then better install protobuf from source into a local folder, say ~/local/;
       Decompress the tar file, and then
       ```
       ./configure --prefix=/home/<yourname>local
       make && make install
       echo "export LD_LIBRARY_PATH=/home/<yourname>/local/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
       source ~/.bashrc
    3. if it cannot find other libs including python, then please create virtual env using pip or conda;
       and then install SINGA via
       ```
       pip install --upgrade <url of singa wheel>
       ```


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

    A: This means the linker found libstdc++.so.6 but that library
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

* Q: When compiling PySINGA from source, there is a compilation error due to the missing of <numpy/objectarray.h>

    A: Please install numpy and export the path of numpy header files as

        $ export CPLUS_INCLUDE_PATH=`python -c "import numpy; print numpy.get_include()"`:$CPLUS_INCLUDE_PATH
