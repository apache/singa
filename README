
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
  * lmdb (OpenLDAP)

##Documentation

Full documentation is available online at [Official Documentation](https://singa.incubator.apache.org/docs/overview.html#).


##Building SINGA
	
	$ ./autogen.sh
	$ ./configure
	$ make

##Running Examples

Let us train the [CNN
model](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) over the
[CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) dataset without parallelism as an example. The hyper-parameters
are set following [cuda-convnet](https://code.google.com/p/cuda-convnet/). More details about this example are available
at [CNN example](http://singa.incubator.apache.org/docs/cnn).

First, download the dataset and create data shards:

	$ cd examples/cifar10/
	$ make download
	$ make create

Next, start the training: 

	$ cd ../../
    	$ ./bin/singa-run.sh -conf examples/cifar10/job.conf

Now we just need to wait until it is done! 

##LICENSE

Apache Singa is licensed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

For additional information, see the LICENSE and NOTICE files.
