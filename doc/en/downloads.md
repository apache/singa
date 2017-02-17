## Download SINGA

* Latest code: please clone the dev branch from [Github](https://github.com/apache/incubator-singa)

* v1.1.0 (12 February 2017):
     * [Apache SINGA 1.1.0](http://www.apache.org/dyn/closer.cgi/incubator/singa/1.1.0/apache-singa-incubating-1.1.0.tar.gz)
      [\[MD5\]](https://dist.apache.org/repos/dist/release/incubator/singa/1.1.0/apache-singa-incubating-1.1.0.tar.gz.md5)
      [\[KEYS\]](https://dist.apache.org/repos/dist/release/incubator/singa/1.1.0/KEYS)
    * [Release Notes 1.1.0](releases/RELEASE_NOTES_1.1.0.html)
    * New features and major updates,
        * Create Docker images (CPU and GPU versions)
        * Create Amazon AMI for SINGA (CPU version)
        * Integrate with Jenkins for automatically generating Wheel and Debian packages (for installation), and updating the website.
        * Enhance the FeedFowardNet, e.g., multiple inputs and verbose mode for debugging
        * Add Concat and Slice layers
        * Extend CrossEntropyLoss to accept instance with multiple labels
        * Add image_tool.py with image augmentation methods
        * Support model loading and saving via the Snapshot API
        * Compile SINGA source on Windows
        * Compile mandatory dependent libraries together with SINGA code
        * Enable Java binding (basic) for SINGA
        * Add version ID in checkpointing files
        * Add Rafiki toolkit for providing RESTFul APIs
        * Add examples pretrained from Caffe, including GoogleNet



* v1.0.0 (8 September 2016):
    * [Apache SINGA 1.0.0](https://archive.apache.org/dist/incubator/singa/1.0.0/apache-singa-incubating-1.0.0.tar.gz)
      [\[MD5\]](https://archive.apache.org/dist/incubator/singa/1.0.0/apache-singa-incubating-1.0.0.tar.gz.md5)
      [\[KEYS\]](https://archive.apache.org/dist/incubator/singa//1.0.0/KEYS)
    * [Release Notes 1.0.0](releases/RELEASE_NOTES_1.0.0.html)
    * New features and major updates,
        * Tensor abstraction for supporting more machine learning models.
        * Device abstraction for running on different hardware devices, including CPU, (Nvidia/AMD) GPU and FPGA (to be tested in later versions).
        * Replace GNU autotool with cmake for compilation.
        * Support Mac OS
        * Improve Python binding, including installation and programming
        * More deep learning models, including VGG and ResNet
        * More IO classes for reading/writing files and encoding/decoding data
        * New network communication components directly based on Socket.
        * Cudnn V5 with Dropout and RNN layers.
        * Replace website building tool from maven to Sphinx
        * Integrate Travis-CI


* v0.3.0 (20 April 2016):
    * [Apache SINGA 0.3.0](https://archive.apache.org/dist/incubator/singa/0.3.0/apache-singa-incubating-0.3.0.tar.gz)
      [\[MD5\]](https://archive.apache.org/dist/incubator/singa/0.3.0/apache-singa-incubating-0.3.0.tar.gz.md5)
      [\[KEYS\]](https://archive.apache.org/dist/incubator/singa/0.3.0/KEYS)
    * [Release Notes 0.3.0](releases/RELEASE_NOTES_0.3.0.html)
    * New features and major updates,
        * [Training on GPU cluster](v0.3.0/gpu.html) enables training of deep learning models over a GPU cluster.
        * [Python wrapper improvement](v0.3.0/python.html) makes it easy to configure the job, including neural net and SGD algorithm.
        * [New SGD updaters](v0.3.0/updater.html) are added, including Adam, AdaDelta and AdaMax.
        * [Installation](v0.3.0/installation.html) has fewer dependent libraries for single node training.
        * Heterogeneous training with CPU and GPU.
        * Support cuDNN V4.
        * Data prefetching.
        * Fix some bugs.



* v0.2.0 (14 January 2016):
    * [Apache SINGA 0.2.0](https://archive.apache.org/dist/incubator/singa/0.2.0/apache-singa-incubating-0.2.0.tar.gz)
      [\[MD5\]](https://archive.apache.org/dist/incubator/singa/0.2.0/apache-singa-incubating-0.2.0.tar.gz.md5)
      [\[KEYS\]](https://archive.apache.org/dist/incubator/singa/0.2.0/KEYS)
    * [Release Notes 0.2.0](releases/RELEASE_NOTES_0.2.0.html)
    * New features and major updates,
        * [Training on GPU](v0.2.0/gpu.html) enables training of complex models on a single node with multiple GPU cards.
        * [Hybrid neural net partitioning](v0.2.0/hybrid.html) supports data and model parallelism at the same time.
        * [Python wrapper](v0.2.0/python.html) makes it easy to configure the job, including neural net and SGD algorithm.
        * [RNN model and BPTT algorithm](v0.2.0/general-rnn.html) are implemented to support applications based on RNN models, e.g., GRU.
        * [Cloud software integration](v0.2.0/distributed-training.html) includes Mesos, Docker and HDFS.
        * Visualization of neural net structure and layer information, which is helpful for debugging.
        * Linear algebra functions and random functions against Blobs and raw data pointers.
        * New layers, including SoftmaxLayer, ArgSortLayer, DummyLayer, RNN layers and cuDNN layers.
        * Update Layer class to carry multiple data/grad Blobs.
        * Extract features and test performance for new data by loading previously trained model parameters.
        * Add Store class for IO operations.


* v0.1.0 (8 October 2015):
    * [Apache SINGA 0.1.0](https://archive.apache.org/dist/incubator/singa/apache-singa-incubating-0.1.0.tar.gz)
      [\[MD5\]](https://archive.apache.org/dist/incubator/singa/apache-singa-incubating-0.1.0.tar.gz.md5)
      [\[KEYS\]](https://archive.apache.org/dist/incubator/singa/KEYS)
    * [Amazon EC2 image](https://console.aws.amazon.com/ec2/v2/home?region=ap-southeast-1#LaunchInstanceWizard:ami=ami-b41001e6)
    * [Release Notes 0.1.0](releases/RELEASE_NOTES_0.1.0.html)
    * Major features include,
        * Installation using GNU build utility
        * Scripts for job management with zookeeper
        * Programming model based on NeuralNet and Layer abstractions.
        * System architecture based on Worker, Server and Stub.
        * Training models from three different model categories, namely, feed-forward models, energy models and RNN models.
        * Synchronous and asynchronous distributed training frameworks using CPU
        * Checkpoint and restore
        * Unit test using gtest

**Disclaimer**

Apache SINGA is an effort undergoing incubation at The Apache Software
Foundation (ASF), sponsored by the name of Apache Incubator PMC. Incubation is
required of all newly accepted projects until a further review indicates that
the infrastructure, communications, and decision making process have stabilized
in a manner consistent with other successful ASF projects. While incubation
status is not necessarily a reflection of the completeness or stability of the
code, it does indicate that the project has yet to be fully endorsed by the
ASF.
