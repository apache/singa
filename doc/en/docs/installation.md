# Installation


## From wheel

Users can download the pre-compiled wheel files to install PySINGA.
PySINGA has been tested on Linux (Ubunu 14.04 and 16.04) and Mac OS (10.11 and 10.12).

### Pre-requisite

Python 2.7 and pip are required

    # For Ubuntu
    $ sudo apt-get install python2.7-dev python-pip

    # For Mac
    $ brew tap homebrew/python
    $ brew install python

Note for Mac OS, you need to configure the (python) paths correctly if multiple python versions are installed.
Refer to FAQ for the errors and solutions.

### Virtual environment

Users are recommended to use PySINGA in python virtual environment.

To use pip with virtual environment,

    # install virtualenv
    $ pip install virtualenv
    $ virtualenv pysinga
    $ source pysinga/bin/activate

To use anaconda with virtual environment,

    $ conda create --name pysinga python=2
    $ source activate pysinga


Note that in python virtual environment, you may need to reset the `PYTHONPATH` to empty
to avoid the conflicts of system path and virtual environment path.


### Instructions

Currently, the following wheel files are available,

<table border="1">
  <tr>
    <th>OS</th>
    <th>Device</th>
    <th>CUDA/cuDNN</th>
    <th>Link</th>
  </tr>
  <tr>
    <td>Ubuntu14.04</td>
    <td>CPU</td>
    <td>-</td>
    <td><a href="http://comp.nus.edu.sg/~dbsystem/singa/assets/file/wheel/linux/latest/ubuntu14.04-cpp/">latest</a>, <a href="http://www.comp.nus.edu.sg/~dbsystem/singa/assets/file/wheel/linux">history</a></td>
  </tr>
  <tr>
    <td>Ubuntu14.04</td>
    <td>GPU</td>
    <td>CUDA7.5+cuDNN4</td>
    <td><a href="http://comp.nus.edu.sg/~dbsystem/singa/assets/file/wheel/linux/latest/ubuntu14.04-cuda7.5-cudnn4/">latest</a>, <a href="http://www.comp.nus.edu.sg/~dbsystem/singa/assets/file/wheel/linux">history</a></td>
  </tr>
  <tr>
    <td>Ubuntu14.04</td>
    <td>GPU</td>
    <td>CUDA7.5+cuDNN5</td>
    <td><a href="http://comp.nus.edu.sg/~dbsystem/singa/assets/file/wheel/linux/latest/ubuntu14.04-cuda7.5-cudnn5/">latest</a>, <a href="http://www.comp.nus.edu.sg/~dbsystem/singa/assets/file/wheel/linux">history</a></td>
  </tr>
  <tr>
    <td>Ubuntu16.04</td>
    <td>CPU</td>
    <td>-</td>
    <td><a href="http://comp.nus.edu.sg/~dbsystem/singa/assets/file/wheel/linux/latest/ubuntu16.04-cpp/">latest</a>, <a href="http://www.comp.nus.edu.sg/~dbsystem/singa/assets/file/wheel/linux">history</a></td>
  </tr>
  <tr>
    <td>Ubuntu16.04</td>
    <td>GPU</td>
    <td>CUDA8.0+cuDNN5</td>
    <td><a href="http://comp.nus.edu.sg/~dbsystem/singa/assets/file/wheel/linux/latest/ubuntu16.04-cuda8.0-cudnn5/">latest</a>, <a href="http://www.comp.nus.edu.sg/~dbsystem/singa/assets/file/wheel/linux">history</a></td>
  </tr>
  <tr>
    <td>MacOSX10.11</td>
    <td>CPU</td>
    <td>-</td>
    <td><a href="http://comp.nus.edu.sg/~dbsystem/singa/assets/file/wheel/macosx/latest/macosx10.11-cpp/">latest</a>, <a href="http://www.comp.nus.edu.sg/~dbsystem/singa/assets/file/wheel/macosx">history</a></td>
  </tr>
  <tr>
    <td>MacOSX10.12</td>
    <td>CPU</td>
    <td>-</td>
    <td><a href="http://comp.nus.edu.sg/~dbsystem/singa/assets/file/wheel/macosx/latest/macosx10.12-cpp/">latest</a>, <a href="http://www.comp.nus.edu.sg/~dbsystem/singa/assets/file/wheel/macosx">history</a></td>
  </tr>
</table>

Download the whl file and execute the following command to install PySINGA,

    $ pip install --upgrade <path to the whel file>

To install the wheel file compiled with CUDA, you need to install CUDA and export the `LD_LIBRARY_PATH` to cuDNN before running the above instruction.

If you have sudo right, you can run the above commands using `sudo pip install` without python virtual environment.
The option `--upgrade` may cause errors sometimes, in which case you can ignore it.

## From Debian Package

The following Debian packages (on architecture: amd64) are available

<table border="1">
  <tr>
    <th>OS</th>
    <th>Device</th>
    <th>CUDA/cuDNN</th>
    <th>Link</th>
  </tr>
  <tr>
    <td>Ubuntu14.04</td>
    <td>CPU</td>
    <td>-</td>
    <td><a href="http://comp.nus.edu.sg/~dbsystem/singa/assets/file/debian/latest/ubuntu14.04-cpp/python-singa.deb">latest</a>, <a href="http://www.comp.nus.edu.sg/~dbsystem/singa/assets/file/debian">history</a></td>
  </tr>
  <tr>
    <td>Ubuntu14.04</td>
    <td>GPU</td>
    <td>CUDA7.5+cuDNN4</td>
    <td>coming soon</td>
  </tr>
  <tr>
    <td>Ubuntu14.04</td>
    <td>GPU</td>
    <td>CUDA7.5+cuDNN5</td>
    <td>coming soon</td>
  </tr>
  <tr>
    <td>Ubuntu16.04</td>
    <td>CPU</td>
    <td>-</td>
    <td><a href="http://comp.nus.edu.sg/~dbsystem/singa/assets/file/debian/latest/ubuntu16.04-cpp/python-singa.deb">latest</a>, <a href="http://www.comp.nus.edu.sg/~dbsystem/singa/assets/file/debian">history</a></td>
  </tr>
  <tr>
    <td>Ubuntu16.04</td>
    <td>GPU</td>
    <td>CUDA8.0+cuDNN5</td>
    <td>coming soon</td>
  </tr>
</table>

Download the deb file and install it via

    apt-get install <path to the deb file, e.g., ./python-singa.deb>

Note that the path must include `./` if the file in inside the current folder.

## From source

The source files could be downloaded either as a [tar.gz file](https://dist.apache.org/repos/dist/dev/incubator/singa/), or as a git repo

    $ git clone https://github.com/apache/incubator-singa.git
    $ cd incubator-singa/

### Pre-requisite

The following libraries are required
* cmake (>=2.8)
* gcc (>=4.8.1) or Clang
* google protobuf (>=2.5,<3)
* blas (tested with openblas >=0.2.10)
* swig(>=3.0.10) for compiling PySINGA
* numpy(>=1.11.0) for compiling PySINGA

The following libraries are optional
* opencv (tested with 2.4.8)
* lmdb (tested with 0.9)
* glog

### Instructions

1. create a `build` folder inside incubator-singa and go into that folder
2. run `cmake [options] ..`
  by default all options are OFF except `USE_PYTHON`

    * `USE_MODUELS=ON`, used if protobuf and blas are not installed a prior
    * `USE_CUDA=ON`, used if CUDA and cuDNN is available
    * `USE_PYTHON=ON`, used for compiling PySINGA
    * `USE_OPENCL=ON`, used for compiling with OpenCL support
3. compile the code, e.g., `make`
4. goto python folder
5. run `pip install .`
6. [optional] run `python setup.py bdist_wheel` to generate the wheel file

Step 4 and 5 are to install PySINGA.
Details on the installation of dependent libraries and the instructions for each OS are given in the following sections.

### Linux and Mac OS

Most of the dependent libraries could be installed from source or via package mangers like
apt-get, yum, and homebrew. Please refer to FAQ for problems caused by the path setting of the dependent libraries.

The following instructions are tested on Ubuntu 14.04  and 16.04for installing dependent libraries.

    # required libraries
    $ sudo apt-get install libprotobuf-dev libopenblas-dev protobuf-compiler

    # optional libraries
    $ sudo apt-get install python2.7-dev python-pip python-numpy
    $ sudo apt-get install libopencv-dev libgoogle-glog-dev liblmdb-dev

The following instructions are tested on Mac OS X Yosemite (10.11 and 10.12) for installing dependent libraries.

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
path,

    $ export CMAKE_INCLUDE_PATH=/usr/local/opt/openblas/include:$CMAKE_INCLUDE_PATH
    $ export CMAKE_LIBRARY_PATH=/usr/local/opt/openblas/lib:$CMAKE_LIBRARY_PATH

To let the runtime know the openblas path,

    $ export LD_LIBRARY_PATH=/usr/local/opt/openblas/library:$LD_LIBRARY_PATH


#### Compile with USE_MODULES=ON

If protobuf and openblas are not installed, you can compile SINGA together with them

    $ In SINGA ROOT folder
    $ mkdir build
    $ cd build
    $ cmake -DUSE_MODULES=ON ..
    $ make

cmake would download OpenBlas and Protobuf (2.6.1) and compile them together
with SINGA.

After compiling SINGA, you can run the unit tests by

    $ ./bin/test_singa

You can see all the testing cases with testing results. If SINGA passes all
tests, then you have successfully installed SINGA.

You can use `ccmake ..` to configure the compilation options.
If some dependent libraries are not in the system default paths, you need to export
the following environment variables

    export CMAKE_INCLUDE_PATH=<path to the header file folder>
    export CMAKE_LIBRARY_PATH=<path to the lib file folder>

#### Compile with USE_PYTHON=ON
swig and numpy can be install by

    $ Ubuntu 14.04 and 16.04
    $ sudo apt-get install python-numpy
    # Ubuntu 16.04
    $ sudo apt-get install swig

Note that swig has to be installed from source on Ubuntu 14.04.
After installing numpy, export the header path of numpy.i as

    $ export CPLUS_INCLUDE_PATH=`python -c "import numpy; print numpy.get_include()"`:$CPLUS_INCLUDE_PATH

Similar to compile CPP code, PySINGA is compiled by

    $ cmake -DUSE_PYTHON=ON ..
    $ make
    $ cd python
    $ pip install .

Developers can build the wheel file via

    # under the build directory
    $ cd python

The generated wheel file is under "dist" directory.


#### Compile SINGA with USE_CUDA=ON

Users are encouraged to install the CUDA and
[cuDNN](https://developer.nvidia.com/cudnn) for running SINGA on GPUs to
get better performance.

SINGA has been tested over CUDA (7, 7.5, 8), and cuDNN (4 and 5).  If cuDNN is
decompressed into non-system folder, e.g. /home/bob/local/cudnn/, the following
commands should be executed for cmake and the runtime to find it

    $ export CMAKE_INCLUDE_PATH=/home/bob/local/cudnn/include:$CMAKE_INCLUDE_PATH
    $ export CMAKE_LIBRARY_PATH=/home/bob/local/cudnn/lib64:$CMAKE_LIBRARY_PATH
    $ export LD_LIBRARY_PATH=/home/bob/local/cudnn/lib64:$LD_LIBRARY_PATH

The cmake options for CUDA and cuDNN should be switched on

    # Dependent libs are install already
    $ cmake -DUSE_CUDA=ON ..

#### Compile SINGA with USE_OPENCL=ON

SINGA uses opencl-headers and viennacl (version 1.7.1 or newer) for OpenCL support, which
can be installed using via

    # On Ubuntu 16.04
    $ sudo apt-get install opencl-headers, libviennacl-dev
    # On Fedora
    $ sudo yum install opencl-headers, viennacl

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

    cmake -DUSE_OPENCL=ON ..

### Compile SINGA on Windows

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
library missing, configure your library file and include path for cmake or the system.
For example, you get an error "Could NOT find CBLAS" and suppose you installed
openblas header files at "d:\include" and openblas library at "d:\lib". You should run the
following command to specify your cblas parameters in cmake:

    $ cmake -G "Visual Studio 14" -DUSE_CUDA=OFF -DUSE_PYTHON=OFF -DCBLAS_INCLUDE_DIR="d:\include" -DCBLAS_LIBRARIES="d:\lib\libopenblas.lib" -DProtobuf_INCLUDE_DIR=<include dir of protobuf> -DProtobuf_LIBRARIES=<path to libprotobuf.lib> -DProtobuf_PROTOC_EXECUTABLE=<path to protoc.exe> -DGLOG_INCLUDE_DIR=<include dir of glog> -DGLOG_LIBRARIES=<path to libglog.lib> ..

To find out the parameters you need to specify for some special libraries, you
can run the following command:

    $ cmake -LAH

If you use cmake GUI tool in windows, make sure you configure the right
parameters for the singa solution by select "Advanced" box. After generating the VS project,
open the "singa.sln" project file under
the "build" folder and compile it as a normal VS solution. You will find the
unit tests file named "test_singa" in the project binary folder.
If you get errors when running test_singa.exe due to libglog.dll/libopenblas.dll missing,
just copy the dll files into the same folder as test_singa.exe

## FAQ

* Q: Error from 'import singa' using PySINGA installed from wheel.

    A: Please check the detailed error from `python -c  "from singa import _singa_wrap"`. Sometimes it is
    caused by the dependent libraries, e.g. there are multiple versions of protobuf or missing of cudnn. Following
    steps show the solutions for different cases
    1. Check the cudnn and cuda and gcc versions, cudnn5 and cuda7.5 and gcc4.8/4.9 are preferred. if gcc is 5.0, then downgrade it.
       If cudnn is missing or not match with the wheel version, you can download the correct version of cudnn into ~/local/cudnn/ and

            $ echo "export LD_LIBRARY_PATH=/home/<yourname>/local/cudnn/lib64:$LD_LIBRARY_PATH" >> ~/.bashrc

    2. If it is the problem related to protobuf, then download the newest whl files which have [compiled protobuf and openblas into the whl](https://issues.apache.org/jira/browse/SINGA-255) file of PySINGA.
       Or you can install protobuf from source into a local folder, say ~/local/;
       Decompress the tar file, and then

            $ ./configure --prefix=/home/<yourname>local
            $ make && make install
            $ echo "export LD_LIBRARY_PATH=/home/<yourname>/local/lib:$LD_LIBRARY_PATH" >> ~/.bashrc
            $ source ~/.bashrc

    3. If it cannot find other libs including python, then create virtual env using pip or conda;
       and then install SINGA via

            $ pip install --upgrade <url of singa wheel>


* Q: Error from running `cmake ..`, which cannot find the dependent libraries.

    A: If you haven't installed the libraries, install them. If you installed
    the libraries in a folder that is outside of the system folder, e.g. /usr/local,
    you need to export the following variables

        $ export CMAKE_INCLUDE_PATH=<path to your header file folder>
        $ export CMAKE_LIBRARY_PATH=<path to your lib file folder>


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

* Q: When I run PySINGA in Mac OS X, I got the error "Fatal Python error: PyThreadState_Get: no current thread  Abort trap: 6"

    A: This error happens typically when you have multiple version of Python on your system,
    e.g, the one comes with the OS and the one installed by Homebrew. The Python linked by PySINGA must be the same as the Python interpreter.
    You can check your interpreter by `which python` and check the Python linked by PySINGA via `otool -L <path to _singa_wrap.so>`.
    To fix this error, compile SINGA with the correct version of Python.
    In particular, if you build PySINGA from source, you need to specify the paths when invoking [cmake](http://stackoverflow.com/questions/15291500/i-have-2-versions-of-python-installed-but-cmake-is-using-older-version-how-do)

        $ cmake -DPYTHON_LIBRARY=`python-config --prefix`/lib/libpython2.7.dylib -DPYTHON_INCLUDE_DIR=`python-config --prefix`/include/python2.7/ ..

    If installed PySINGA from binary packages, e.g. debian or wheel, then you need to change the python interpreter, e.g., reset the $PATH to put the correct path of Python at the front position.
