#singa-incubating-0.2.0 Release Notes

---

SINGA is a general distributed deep learning platform for training big deep
learning models over large datasets. It is designed with an intuitive
programming model based on the layer abstraction. SINGA supports a wide variety
of popular deep learning models.

This release includes the following **major features**:

* [Training on GPU](../docs/gpu.html) enables training of complex models on a single node with multiple GPU cards.
* [Hybrid neural net partitioning](../docs/hybrid.html) supports data and model parallelism at the same time.
* [Python wrapper](../docs/python.html) makes it easy to configure the job, including neural net and SGD algorithm.
* [RNN model and BPTT algorithm](../docs/general-rnn.html) are implemented to support applications based on RNN models, e.g., GRU.
* [Cloud software integration](../docs/distributed-training.md) includes Mesos, Docker and HDFS.


**More details** are listed as follows,

  * Programming model
    * [SINGA-80] New Blob Level and Address Level Math Operation Interface
    * [SINGA-82] Refactor input layers using data store abstraction
    * [SINGA-87] Replace exclude field to include field for layer configuration
    * [SINGA-110] Add Layer member datavec_ and gradvec_
    * [SINGA-120] Implemented GRU and BPTT (BPTTWorker)


  * Neuralnet layers
    * [SINGA-91] Add SoftmaxLayer and ArgSortLayer
    * [SINGA-106] Add dummy layer for test purpose
    * [SINGA-120] Implemented GRU and BPTT (GRULayer and OneHotLayer)


  * GPU training support
    * [SINGA-100] Implement layers using CUDNN for GPU training
    * [SINGA-104] Add Context Class
    * [SINGA-105] Update GUN make files for compiling cuda related code
    * [SINGA-98] Add Support for AlexNet ImageNet Classification Model


  * Model/Hybrid partition
    * [SINGA-109] Refine bridge layers
    * [SINGA-111] Add slice, concate and split layers
    * [SINGA-113] Model/Hybrid Partition Support


  * Python binding
    * [SINGA-108] Add Python wrapper to singa


  * Predict-only mode
    * [SINGA-85] Add functions for extracting features and test new data


  * Integrate with third-party tools
    * [SINGA-11] Start SINGA on Apache Mesos
    * [SINGA-78] Use Doxygen to generate documentation
    * [SINGA-89] Add Docker support


  * Unit test
    * [SINGA-95] Add make test after building


  * Other improvment
    * [SINGA-84] Header Files Rearrange
    * [SINGA-93] Remove the asterisk in the log tcp://169.254.12.152:*:49152
    * [SINGA-94] Move call to google::InitGoogleLogging() from Driver::Init() to main()
    * [SINGA-96] Add Momentum to Cifar10 Example
    * [SINGA-101] Add ll (ls -l) command in .bashrc file when using docker
    * [SINGA-114] Remove short logs in tmp directory
    * [SINGA-115] Print layer debug information in the neural net graph file
    * [SINGA-118] Make protobuf LayerType field id easy to assign
    * [SIGNA-97] Add HDFS Store


  * Bugs fixed
    * [SINGA-85] Fix compilation errors in examples
    * [SINGA-90] Miscellaneous trivial bug fixes
    * [SINGA-107] Error from loading pre-trained params for training stacked RBMs
    * [SINGA-116] Fix a bug in InnerProductLayer caused by weight matrix sharing


