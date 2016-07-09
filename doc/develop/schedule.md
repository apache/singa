
# Development Schedule

| Release | Module| Feature | Status |
|---------|---------|-------------|--------|
| 0.1 Sep 2015     | Neural Network | Feed forward neural network, including CNN, MLP | done|
|         |          | RBM-like model, including RBM | done|
|         |                | Recurrent neural network, including standard RNN | done|
|         | Architecture   | One worker group on single node (with data partition)| done|
|         |                | Multi worker groups on single node using [Hogwild](http://www.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf)|done|
|         |                | Distributed Hogwild|done|
|         |                | Multi groups across nodes, like [Downpour](http://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks)|done|
|         |                | All-Reduce training architecture like [DeepImage](http://arxiv.org/abs/1501.02876)|done|
|         |                | Load-balance among servers | done|
|         | Failure recovery| Checkpoint and restore |done|
|         | Tools| Installation with GNU auto tools| done|
|0.2 Jan 2016 | Neural Network | Feed forward neural network, including AlexNet, cuDNN layers, etc.| done |
|         |                | Recurrent neural network, including GRULayer and BPTT|done |
|         | | Model partition and hybrid partition|done|
|         | Tools | Integration with Mesos for resource management|done|
|         |               | Prepare Docker images for deployment|done|
|         |               | Visualization of neural net and debug information |done|
|         | Binding        | Python binding for major components |done|
|         | GPU            | Single node with multiple GPUs |done|
|0.3 April 2016 | GPU | Multiple nodes, each with multiple GPUs|done|
|               |     | Heterogeneous training using both GPU and CPU [CcT](http://arxiv.org/abs/1504.04343)|done|
|               |     | Support cuDNN v4 | done|
|               | Installation| Remove dependency on ZeroMQ, CZMQ, Zookeeper for single node training|done|
|               | Updater| Add new SGD updaters including Adam, AdamMax and AdaDelta|done|
|               | Binding| Enhance Python binding for training|done|
|0.4 June 2016  | Rafiki | Deep learning as a service| |
|               |        | Product search using Rafiki| |
|1.0 July 2016  | Programming abstraction|Tensor with linear algebra, neural net and random operations| |
|               |                        |Updater for distributed parameter updating||
|               | Optimization       | Execution and memory optimization||
|               | Hardware           | Use Cuda and Cudnn for Nvidia GPU||
|               |                    | Use OpenCL for AMD GPU or other devices||
|               | Cross-platform | To extend from Linux to MacOS and Windows||
|               | Examples | Speech recognition example||
|               | |Large image models, e.g., [GoogLeNet](http://arxiv.org/abs/1409.4842), [VGG](https://arxiv.org/pdf/1409.1556.pdf) and [Residual Net](http://arxiv.org/abs/1512.03385)||
