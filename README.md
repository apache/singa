#Apache SINGA

Distributed deep learning system

##Project Website

All the details can be found in [Project Website](http://singa.incubator.apache.org), including the following instructions.

##Mailing Lists

* [Development Mailing List](mailto:dev-subscribe@singa.incubator.apache.org)([Archive](http://mail-archives.apache.org/mod_mbox/singa-dev/))
* [Commits Mailing List](mailto:commits-subscribe@singa.incubator.apache.org)([Archive](http://mail-archives.apache.org/mod_mbox/singa-commits/))

##Dependencies
The current code depends on the following external libraries:

  * glog (New BSD)
  * google-protobuf (New BSD)
  * openblas (New BSD)
  * zeromq (LGPLv3 + static link exception)
  * czmq (Mozilla Public License Version 2.0)
  * zookeeper (Apache 2.0)

You can install all dependencies into $PREFIX folder by

    ./thirdparty/install.sh all $PREFIX

If $PREFIX is not a system path (e.g., /usr/local/), you have to export some
environment variables,

    export LD_LIBRARY_PATH=$PREFIX/lib:$LD_LIBRARY_PATH
    export CPLUS_INCLUDE_PATH=$PREFIX/include

##Documentation

Full documentation is available online at [Official Documentation](https://singa.incubator.apache.org/docs/overview.html#).

##Building SINGA

    $ ./autogen.sh (optional)
    # pls refer to FAQ for solutions of errors
    $ ./configure
    $ make

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

If it reports errors due to libopenblas.so missing, then include the
lib folder of OpenBLAS in LD_LIBRARY_PATH

    $ export LD_LIBRARY_PATH=$OPENBLAS_FOLDER/lib:$LD_LIBRARY_PATH
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

For additional information, see the LICENSE and NOTICE files.

## FAQ

* Q1:I get error `./configure --> cannot find blas_segmm() function` even I
run `install.sh OpenBLAS`.

  A1: `OpenBLAS` library is installed in `/opt` folder by default or
  other folders if you use `sudo apt-get install`.
  You need to include the OpenBLAS library folder in the LDFLAGS, e.g.,

      $ export LDFLAGS=-L/opt/OpenBLAS

  Alternatively, you can include the path in LIBRARY_PATH.


* Q2: I get error `cblas.h no such file or directory exists`.

  Q2: You need to include the folder of the cblas.h into CPLUS_INCLUDE_PATH,
  e.g.,

      $ export CPLUS_INCLUDE_PATH=/opt/OpenBLAS/include:$CPLUS_INCLUDE_PATH
      # reconfigure and make SINGA
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

	  $ sudo apt-get install openblas

  or

	  $ sudo yum install openblas-devel

  It is worth noting that you need root access to run the last two commands.
