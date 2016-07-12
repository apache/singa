#singa-incubating-0.1.0 Release Notes

---

SINGA is a general distributed deep learning platform for training big deep learning models over large datasets. It is
designed with an intuitive programming model based on the layer abstraction. SINGA supports a wide variety of popular
deep learning models.

This release includes following features:

  * Job management
    * [SINGA-3](https://issues.apache.org/jira/browse/SINGA-3)  Use Zookeeper to check stopping (finish) time of the system
    * [SINGA-16](https://issues.apache.org/jira/browse/SINGA-16)  Runtime Process id Management
    * [SINGA-25](https://issues.apache.org/jira/browse/SINGA-25)  Setup glog output path
    * [SINGA-26](https://issues.apache.org/jira/browse/SINGA-26)  Run distributed training in a single command
    * [SINGA-30](https://issues.apache.org/jira/browse/SINGA-30)  Enhance easy-to-use feature and support concurrent jobs
    * [SINGA-33](https://issues.apache.org/jira/browse/SINGA-33)  Automatically launch a number of processes in the cluster
    * [SINGA-34](https://issues.apache.org/jira/browse/SINGA-34)  Support external zookeeper service
    * [SINGA-38](https://issues.apache.org/jira/browse/SINGA-38)  Support concurrent jobs
    * [SINGA-39](https://issues.apache.org/jira/browse/SINGA-39)  Avoid ssh in scripts for single node environment
    * [SINGA-43](https://issues.apache.org/jira/browse/SINGA-43)  Remove Job-related output from workspace
    * [SINGA-56](https://issues.apache.org/jira/browse/SINGA-56)  No automatic launching of zookeeper service
    * [SINGA-73](https://issues.apache.org/jira/browse/SINGA-73)  Refine the selection of available hosts from host list


  * Installation with GNU Auto tool
    * [SINGA-4](https://issues.apache.org/jira/browse/SINGA-4)  Refine thirdparty-dependency installation
    * [SINGA-13](https://issues.apache.org/jira/browse/SINGA-13)  Separate intermediate files of compilation from source files
    * [SINGA-17](https://issues.apache.org/jira/browse/SINGA-17)  Add root permission within thirdparty/install.
    * [SINGA-27](https://issues.apache.org/jira/browse/SINGA-27)  Generate python modules for proto objects
    * [SINGA-53](https://issues.apache.org/jira/browse/SINGA-53)  Add lmdb compiling options
    * [SINGA-62](https://issues.apache.org/jira/browse/SINGA-62)  Remove building scrips and auxiliary files
    * [SINGA-67](https://issues.apache.org/jira/browse/SINGA-67)  Add singatest into build targets


  * Distributed training
    * [SINGA-7](https://issues.apache.org/jira/browse/SINGA-7)  Implement shared memory Hogwild algorithm
    * [SINGA-8](https://issues.apache.org/jira/browse/SINGA-8)  Implement distributed Hogwild
    * [SINGA-19](https://issues.apache.org/jira/browse/SINGA-19)  Slice large Param objects for load-balance
    * [SINGA-29](https://issues.apache.org/jira/browse/SINGA-29)  Update NeuralNet class to enable layer partition type customization
    * [SINGA-24](https://issues.apache.org/jira/browse/SINGA-24)  Implement Downpour training framework
    * [SINGA-32](https://issues.apache.org/jira/browse/SINGA-32)  Implement AllReduce training framework
    * [SINGA-57](https://issues.apache.org/jira/browse/SINGA-57)  Improve Distributed Hogwild


  * Training algorithms for different model categories
    * [SINGA-9](https://issues.apache.org/jira/browse/SINGA-9)  Add Support for Restricted Boltzman Machine (RBM) model
    * [SINGA-10](https://issues.apache.org/jira/browse/SINGA-10)  Add Support for Recurrent Neural Networks (RNN)


  * Checkpoint and restore
    * [SINGA-12](https://issues.apache.org/jira/browse/SINGA-12)  Support Checkpoint and Restore


  * Unit test
    * [SINGA-64](https://issues.apache.org/jira/browse/SINGA-64)  Add the test module for utils/common


  * Programming model
    * [SINGA-36](https://issues.apache.org/jira/browse/SINGA-36)  Refactor job configuration, driver program and scripts
    * [SINGA-37](https://issues.apache.org/jira/browse/SINGA-37)  Enable users to set parameter sharing in model configuration
    * [SINGA-54](https://issues.apache.org/jira/browse/SINGA-54)  Refactor job configuration to move fields in ModelProto out
    * [SINGA-55](https://issues.apache.org/jira/browse/SINGA-55)  Refactor main.cc and singa.h
    * [SINGA-61](https://issues.apache.org/jira/browse/SINGA-61)  Support user defined classes
    * [SINGA-65](https://issues.apache.org/jira/browse/SINGA-65)  Add an example of writing user-defined layers


  * Other features
    * [SINGA-6](https://issues.apache.org/jira/browse/SINGA-6)  Implement thread-safe singleton
    * [SINGA-18](https://issues.apache.org/jira/browse/SINGA-18)  Update API for displaying performance metric
    * [SINGA-77](https://issues.apache.org/jira/browse/SINGA-77)  Integrate with Apache RAT


Some bugs are fixed during the development of this release

  * [SINGA-2](https://issues.apache.org/jira/browse/SINGA-2) Check failed: zsock_connect
  * [SINGA-5](https://issues.apache.org/jira/browse/SINGA-5) Server early terminate when zookeeper singa folder is not initially empty
  * [SINGA-15](https://issues.apache.org/jira/browse/SINGA-15) Fixg a bug from ConnectStub function which gets stuck for connecting layer_dealer_
  * [SINGA-22](https://issues.apache.org/jira/browse/SINGA-22) Cannot find openblas library when it is installed in default path
  * [SINGA-23](https://issues.apache.org/jira/browse/SINGA-23) Libtool version mismatch error.
  * [SINGA-28](https://issues.apache.org/jira/browse/SINGA-28) Fix a bug from topology sort of Graph
  * [SINGA-42](https://issues.apache.org/jira/browse/SINGA-42) Issue when loading checkpoints
  * [SINGA-44](https://issues.apache.org/jira/browse/SINGA-44) A bug when reseting metric values
  * [SINGA-46](https://issues.apache.org/jira/browse/SINGA-46) Fix a bug in updater.cc to scale the gradients
  * [SINGA-47](https://issues.apache.org/jira/browse/SINGA-47) Fix a bug in data layers that leads to out-of-memory when group size is too large
  * [SINGA-48](https://issues.apache.org/jira/browse/SINGA-48) Fix a bug in trainer.cc that assigns the same NeuralNet instance to workers from diff groups
  * [SINGA-49](https://issues.apache.org/jira/browse/SINGA-49) Fix a bug in HandlePutMsg func that sets param fields to invalid values
  * [SINGA-66](https://issues.apache.org/jira/browse/SINGA-66) Fix bugs in Worker::RunOneBatch function and ClusterProto
  * [SINGA-79](https://issues.apache.org/jira/browse/SINGA-79) Fix bug in singatool that can not parse -conf flag


Features planned for the next release

  * [SINGA-11](https://issues.apache.org/jira/browse/SINGA-11) Start SINGA using Mesos
  * [SINGA-31](https://issues.apache.org/jira/browse/SINGA-31) Extend Blob to support xpu (cpu or gpu)
  * [SINGA-35](https://issues.apache.org/jira/browse/SINGA-35) Add random number generators
  * [SINGA-40](https://issues.apache.org/jira/browse/SINGA-40) Support sparse Param update
  * [SINGA-41](https://issues.apache.org/jira/browse/SINGA-41) Support single node single GPU training

