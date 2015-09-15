
#Apache SINGA

Distributed deep learning system

##Project Website

All the details can be found in [Project Website](http://singa.incubator.apache.org), including the following instructions.

##Mailing Lists

* [Development Mailing List](mailto:dev-subscribe@singa.incubator.apache.org)([Archive](http://mail-archives.apache.org/mod_mbox/singa-dev/))
* [Commits Mailing List](mailto:commits-subscribe@singa.incubator.apache.org)([Archive](http://mail-archives.apache.org/mod_mbox/singa-commits/))

##Documentation

Documentation is available in [Official Documentation](https://singa.incubator.apache.org/docs/overview.html#).

##Building SINGA

Just clone our github repo and execute following commands:

	$ git clone git@github.com:apache/incubator-singa.git
	$ cd incubator-singa
	$ ./autogen.sh
	$ ./configure
	$ make

You can also download the release package from the [Project Website](http://singa.incubator.apache.org) and follow the instructions on [Official Installation Guide](http://singa.incubator.apache.org/docs/installation.html).

##Running an Example

After installation, you may want to run an example to try SINGA.
Let us train the [CNN model](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks) over the
[CIFAR-10](http://www.cs.toronto.edu/~kriz/cifar.html) dataset without parallelism as an instance.
The hyper-parameters are set following
[cuda-convnet](https://code.google.com/p/cuda-convnet/). More details about this example are
available at [CNN example](http://singa.incubator.apache.org/docs/cnn).

We firstly download the dataset and create the data shard:

	$ cd examples/cifar10/
	$ make download
	$ make create

Then we start training:

	$ cd ../../
    $ ./bin/singa-run.sh -conf examples/cifar10/job.conf

Now we just need to wait until it is done! About this part, you can also refer to the [Quick Start](http://singa.incubator.apache.org/docs/quick-start.html). 

##LICENSE

Apache Singa is licensed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).

For additional information, see the LICENSE and NOTICE files.
