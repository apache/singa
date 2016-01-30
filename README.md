#Apache SINGA

Distributed deep learning system

##Project Website

All the details can be found in [Project Website](http://singa.incubator.apache.org), including the following instructions.

##Mailing Lists

* [Development Mailing List](mailto:dev-subscribe@singa.incubator.apache.org) ([Archive](http://mail-archives.apache.org/mod_mbox/singa-dev/))
* [Commits Mailing List](mailto:commits-subscribe@singa.incubator.apache.org) ([Archive](http://mail-archives.apache.org/mod_mbox/singa-commits/))

<a name="Dependencies"</a>
##Dependencies
The current code depends on the following external libraries:

  * `glog` (New BSD)
  * `google-protobuf` (New BSD)
  * `openblas` (New BSD)
  * `zeromq` (LGPLv3 + static link exception)
  * `czmq` (Mozilla Public License Version 2.0)
  * `zookeeper` (Apache 2.0)

We have tested SINGA on Ubuntu 12.04, Ubuntu 14.01 and CentOS 6.
You can install all dependencies into `$PREFIX` folder by

    ./thirdparty/install.sh all $PREFIX

If `$PREFIX` is not a system path (e.g., `/usr/local/`), please export the following
variables to continue the building instructions,

    $ export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH
    $ export CPLUS_INCLUDE_PATH=$PREFIX/include:$CPLUS_INCLUDE_PATH
    $ export LIBRARY_PATH=$PREFIX/lib:$LIBRARY_PATH
    $ export PATH=$PREFIX/bin:$PATH

###Optional dependencies
For advanced features, the following libraries are needed:

  * `cuda` (NIVIDA CUDA Toolkit EUL)
  * `cudnn` (NIVIDA CuDNN EULA)
  * `Apache Mesos` (Apache 2.0)
  * `Apache Hadoop` (Apache 2.0)
  * `libhdfs3` (Apache 2.0)
  * `swig` (GPL)

##Documentation

Full documentation is available online at [Official Documentation](https://singa.incubator.apache.org/docs/overview.html).

##Building SINGA

Please make sure you have `g++ >= 4.8.1` before building SINGA.

    $ ./autogen.sh
    # refer to the FAQs below for errors during configure, including blas_segmm() error
    $ ./configure
    # refer to the FAQs below for error during make
    $ make

To compile with GPU support, you should run:

    $ ./configure --enable-cuda --with-cuda=/CUDA/PATH --enable-cudnn --with-cudnn=/CUDNN/PATH

--with-cuda and --with-cudnn are optional as by default the script will search system paths.
Please kindly set proper environment parameters (LD_LIBRARY_PATH, LIBRARY_PATH, etc.) when you run the code.

To compile with HDFS support, you should run:

    $ ./configure --enable-hdfs --with-libhdfs=/PATH/TO/HDFS3

--with-libhdfs is optional as by default the path is /usr/local/.

To compile with python wrappers, you should run:

	$ ./tool/python/singa/generatepy.sh
	$ ./configure --enable-python --with-python=/PATH/TO/Python.h

--with-python is optinal as by default the path is /usr/local/include.

You can also run the following command for further configuration.

    $ ./configure --help

##Running Examples

Let us train the [CNN model](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) over the
[CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) dataset without parallelism as an example. The hyper-parameters
are set following [cuda-convnet](https://code.google.com/p/cuda-convnet/). More details about this example are available
at [CNN example](http://singa.incubator.apache.org/docs/cnn).

First, download the dataset and create data shards:

    $ cd examples/cifar10/
    $ cp Makefile.example Makefile
    $ make download
    $ make create

If it reports errors due to library missing, e.g., `libopenblas` or `libprotobuf`,
please export the environment variables shown in the [Dependencies](#Dependencies) section and
continue with the following instructions,

    # delete the newly created folders
    $ rm -rf cifar10_t*
    $ make create

Next, start the training:

    $ cd ../../
    $ ./bin/zk-service.sh start
    $ ./bin/singa-run.sh -conf examples/cifar10/job.conf

Now we just need to wait until it is done!

##LICENSE

Apache SINGA is licensed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

For additional information, see the `LICENSE` and `NOTICE` files.

## FAQ

* Q1:I get error `./configure --> cannot find blas_segmm() function` even I
have installed `OpenBLAS`.

  A1: This means the compiler cannot find the `OpenBLAS` library. If you have installed `OpenBLAS` via `apt-get install`, then export the path to `$LD_LIBRARY_PATH` (e.g. `/usr/lib/openblas-base`). If you installed it with
  `./thirdparty/install.sh`, then export the correct path based on `$PREFIX` (e.g. `/opt/OpenBLAS/lib`):

      # using apt-get install for openblas
      $ export LIBRARY_PATH=$PATH_TO_OPENBLAS_LIB:$LIBRARY_PATH

      # using ./thirdparty/install.sh for openblas:
      $ export LIBRARY_PATH=/opt/OpenBLAS/lib:$LIBRARY_PATH


* Q2: I get error `cblas.h no such file or directory exists`.

  A2: You need to include the folder containing `cblas.h` into `$CPLUS_INCLUDE_PATH`,
  e.g.,

      $ export CPLUS_INCLUDE_PATH=$PREFIX/include:$CPLUS_INCLUDE_PATH
      # e.g.,
      $ export CPLUS_INCLUDE_PATH=/opt/OpenBLAS/include:$CPLUS_INCLUDE_PATH
      # then reconfigure and make SINGA
      $ ./configure
      $ make


* Q3: When compiling, I get error `SSE2 instruction set not enabled`

  A3: You can try following command:

      $ make CFLAGS='-msse2' CXXFLAGS='-msse2'


* Q4: I get `ImportError: cannot import name enum_type_wrapper` from
`google.protobuf.internal` when I try to import `.py` files.

  A4: After installing `protobuf` by `make install`, we should install `python`
  runtime libraries. Go to `protobuf` source directory, run:

      $ cd /PROTOBUF/SOURCE/FOLDER
      $ cd python
      $ python setup.py build
      $ python setup.py install

  You may need `sudo` when you try to install `python` runtime libraries in
  the system folder.


* Q5: I get a linking error caused by `gflags`.

  A5: SINGA does not depend on `gflags`. But you may have installed the `glog` with
  `gflags`. In that case you can reinstall `glog` using `thirdparty/install.sh` into
  a another folder and export the `$LDFLAGS` and `$CPPFLAGS` to include that folder.


* Q6: While compiling SINGA and installing `glog` on mac OS X, I get fatal error
`'ext/slist' file not found`

  A6: Please install `glog` individually and try :

      $ make CFLAGS='-stdlib=libstdc++' CXXFLAGS='stdlib=libstdc++'

* Q7: When I start a training job, it reports error related to `ZOO_ERROR...zk retcode=-4...`.

  A7: This is because `zookeeper` is not started. Please start the service

      $ ./bin/zk-service.sh start

  If the error still exists, probably that you do not have `java`. You can simply
  check it by

      $ java --version

* Q8: When I build `OpenBLAS` from source, I am told that I need a fortran compiler.

  A8: You can compile `OpenBLAS` by

      $ make ONLY_CBLAS=1

  or install it using

      $ sudo apt-get install openblas-dev

  or

      $ sudo yum install openblas-devel

  It is worth noting that you need root access to run the last two commands.
  Remember to set the environment variables to include the header and library
  paths of `OpenBLAS` after installation (please refer to the [Dependencies](#Dependencies) section).

* Q9: When I build protocol buffer, it reports that `GLIBC++_3.4.20 not found in /usr/lib64/libstdc++.so.6`.

  A9: This means the linker found `libstdc++.so.6` but that library
  belongs to an older version of `GCC` than was used to compile and link the
  program. The program depends on code defined in
  the newer `libstdc++` that belongs to the newer version of GCC, so the linker
  must be told how to find the newer `libstdc++` shared library.
  The simplest way to fix this is to find the correct `libstdc++` and export it to
  `$LD_LIBRARY_PATH`. For example, if `GLIBC++_3.4.20` is listed in the output of the
  following command,

      $ strings /usr/local/lib64/libstdc++.so.6|grep GLIBC++

  then just set your environment variable as

      $ export LD_LIBRARY_PATH=/usr/local/lib64:$LD_LIBRARY_PATH
