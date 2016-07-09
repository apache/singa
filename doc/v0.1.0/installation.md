# Installation

---

## Dependencies

SINGA is developed and tested on Linux platforms.

The following dependent libraries are required:

  * glog version 0.3.3

  * google-protobuf version 2.6.0

  * openblas version >= 0.2.10

  * zeromq version >= 3.2

  * czmq version >= 3

  * zookeeper version 3.4.6


Optional dependencies include:

  * lmdb version 0.9.10


You can install all dependencies into $PREFIX folder by

    ./thirdparty/install.sh all $PREFIX

If $PREFIX is not a system path (e.g., /usr/local/), please export the following
variables to continue the building instructions,

    export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH
    export CPLUS_INCLUDE_PATH=$PREFIX/include:$CPLUS_INCLUDE_PATH
    export LIBRARY_PATH=$PREFIX/lib:$LIBRARY_PATH
    export PATH=$PREFIX/bin:$PATH

More details on using this script is given below.

## Building SINGA from source

SINGA is built using GNU autotools. GCC (version >= 4.8) is required.
There are two ways to build SINGA,

  * If you want to use the latest code, please clone it from
  [Github](https://github.com/apache/incubator-singa.git) and execute
  the following commands,

        $ git clone git@github.com:apache/incubator-singa.git
        $ cd incubator-singa
        $ ./autogen.sh
        $ ./configure
        $ make

  Note: It is an oversight that we forgot to delete the singa repo under [nusinga](https://github.com/orgs/nusinga)
  account after we became Apache Incubator project -- the source
  in that repo was not up to date, and we apologize for any inconvenience.

  * If you download a release package, please follow the instructions below,

        $ tar xvf singa-xxx
        $ cd singa-xxx
        $ ./configure
        $ make

    Some features of SINGA depend on external libraries. These features can be
    compiled with `--enable-<feature>`.
    For example, to build SINGA with lmdb support, you can run:

        $ ./configure --enable-lmdb

<!---
Zhongle: please update the code to use the follow command

    $ make test

After compilation, you will find the binary file singatest. Just run it!
More details about configure script can be found by running:

		$ ./configure -h
-->

After compiling SINGA successfully, the *libsinga.so* and the executable file
*singa* will be generated into *.libs/* folder.

If some dependent libraries are missing (or not detected), you can use the
following script to download and install them:

<!---
to be updated after zhongle changes the code to use

    ./install.sh libname \-\-prefix=

-->

    $ cd thirdparty
    $ ./install.sh LIB_NAME PREFIX

If you do not specify the installation path, the library will be installed in
the default folder specified by the software itself.  For example, if you want
to install `zeromq` library in the default system folder, run it as

    $ ./install.sh zeromq

Or, if you want to install it into another folder,

    $ ./install.sh zeromq PREFIX

You can also install all dependencies in */usr/local* directory:

    $ ./install.sh all /usr/local

Here is a table showing the first arguments:

    LIB_NAME  LIBRARIE
    czmq*                 czmq lib
    glog                  glog lib
    lmdb                  lmdb lib
    OpenBLAS              OpenBLAS lib
    protobuf              Google protobuf
    zeromq                zeromq lib
    zookeeper             Apache zookeeper

*: Since `czmq` depends on `zeromq`, the script offers you one more argument to
indicate `zeromq` location.
The installation commands of `czmq` is:

<!---
to be updated to

    $./install.sh czmq  \-\-prefix=/usr/local \-\-zeromq=/usr/local/zeromq
-->

    $./install.sh czmq  /usr/local -f=/usr/local/zeromq

After the execution, `czmq` will be installed in */usr/local*. The last path
specifies the path to zeromq.

### FAQ
* Q1:I get error `./configure --> cannot find blas_segmm() function` even I
have installed OpenBLAS.

  A1: This means the compiler cannot find the `OpenBLAS` library. If you installed
  it to $PREFIX (e.g., /opt/OpenBLAS), then you need to export it as

      $ export LIBRARY_PATH=$PREFIX/lib:$LIBRARY_PATH
      # e.g.,
      $ export LIBRARY_PATH=/opt/OpenBLAS/lib:$LIBRARY_PATH


* Q2: I get error `cblas.h no such file or directory exists`.

  Q2: You need to include the folder of the cblas.h into CPLUS_INCLUDE_PATH,
  e.g.,

      $ export CPLUS_INCLUDE_PATH=$PREFIX/include:$CPLUS_INCLUDE_PATH
      # e.g.,
      $ export CPLUS_INCLUDE_PATH=/opt/OpenBLAS/include:$CPLUS_INCLUDE_PATH
      # then reconfigure and make SINGA
      $ ./configure
      $ make


* Q3:While compiling SINGA, I get error `SSE2 instruction set not enabled`

  A3:You can try following command:

      $ make CFLAGS='-msse2' CXXFLAGS='-msse2'


* Q4:I get `ImportError: cannot import name enum_type_wrapper` from
google.protobuf.internal when I try to import .py files.

  A4:After install google protobuf by `make install`, we should install python
  runtime libraries. Go to protobuf source directory, run:

      $ cd /PROTOBUF/SOURCE/FOLDER
      $ cd python
      $ python setup.py build
      $ python setup.py install

  You may need `sudo` when you try to install python runtime libraries in
  the system folder.


* Q5: I get a linking error caused by gflags.

  A5: SINGA does not depend on gflags. But you may have installed the glog with
  gflags. In that case you can reinstall glog using *thirdparty/install.sh* into
  a another folder and export the LDFLAGS and CPPFLAGS to include that folder.


* Q6: While compiling SINGA and installing `glog` on mac OS X, I get fatal error
`'ext/slist' file not found`

  A6:Please install `glog` individually and try :

      $ make CFLAGS='-stdlib=libstdc++' CXXFLAGS='stdlib=libstdc++'

* Q7: When I start a training job, it reports error related with "ZOO_ERROR...zk retcode=-4...".

  A7: This is because the zookeeper is not started. Please start the zookeeper service

      $ ./bin/zk-service start

  If the error still exists, probably that you do not have java. You can simple
  check it by

      $ java --version

* Q8: When I build OpenBLAS from source, I am told that I need a fortran compiler.

  A8: You can compile OpenBLAS by

      $ make ONLY_CBLAS=1

  or install it using

	    $ sudo apt-get install openblas-dev

  or

	    $ sudo yum install openblas-devel

  It is worth noting that you need root access to run the last two commands.
  Remember to set the environment variables to include the header and library
  paths of OpenBLAS after installation (please refer to the Dependencies section).

* Q9: When I build protocol buffer, it reports that GLIBC++_3.4.20 not found in /usr/lib64/libstdc++.so.6.

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
