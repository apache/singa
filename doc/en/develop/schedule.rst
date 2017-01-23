.. Licensed to the Apache Software Foundation (ASF) under one
   or more contributor license agreements.  See the NOTICE file
   distributed with this work for additional information
   regarding copyright ownership.  The ASF licenses this file
   to you under the Apache License, Version 2.0 (the
   "License"); you may not use this file except in compliance
   with the License.  You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing,
   software distributed under the License is distributed on an
   "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
   KIND, either express or implied.  See the License for the
   specific language governing permissions and limitations
   under the License.


Development Schedule
====================

.. csv-table::
  :header: "Release","Module","Feature"

  "v0.1 Sep 2015      ","Neural Network               ","Feed forward neural network, including CNN, MLP                                                                     "
  "                   ","                             ","RBM-like model, including RBM                                                                                       "
  "                   ","                             ","Recurrent neural network, including standard RNN                                                                    "
  "                   ","Architecture                 ","One worker group on single node (with data partition)                                                               "
  "                   ","                             ","Multi worker groups on single node using `Hogwild <http://www.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf>`_     "
  "                   ","                             ","Distributed Hogwild"
  "                   ","                             ","Multi groups across nodes, like `Downpour <http://papers.nips.cc/paper/4687-large-scale-distributed-deep-networks>`_"
  "                   ","                             ","All-Reduce training architecture like `DeepImage <http://arxiv.org/abs/1501.02876>`_                                "
  "                   ","                             ","Load-balance among servers                                                                                          "
  "                   ","Failure recovery             ","Checkpoint and restore                                                                                              "
  "                   ","Tools                        ","Installation with GNU auto Tools                                                                                    "
  "v0.2 Jan 2016      ","Neural Network               ","Feed forward neural network, including AlexNet, cuDNN layers,Tools                                                  "
  "                   ","                             ","Recurrent neural network, including GRULayer and BPTT                                                               "
  "                   ","                             ","Model partition and hybrid partition                                                                                "
  "                   ","Tools                        ","Integration with Mesos for resource management                                                                      "
  "                   ","                             ","Prepare Docker images for deployment"
  "                   ","                             ","Visualization of neural net and debug information "
  "                   ","Binding                      ","Python binding for major components "
  "                   ","GPU                          ","Single node with multiple GPUs "
  "v0.3 April 2016    ","GPU                          ","Multiple nodes, each with multiple GPUs"
  "                   ","                             ","Heterogeneous training using both GPU and CPU `CcT <http://arxiv.org/abs/1504.04343>`_"
  "                   ","                             ","Support cuDNN v4 "
  "                   ","Installation                 ","Remove dependency on ZeroMQ, CZMQ, Zookeeper for single node training"
  "                   ","Updater                      ","Add new SGD updaters including Adam, AdamMax and AdaDelta"
  "                   ","Binding                      ","Enhance Python binding for training"
  "v1.0 Sep 2016      ","Programming abstraction      ","Tensor with linear algebra, neural net and random operations "
  "                   ","                             ","Updater for distributed parameter updating "
  "                   ","Hardware                     ","Use Cuda and Cudnn for Nvidia GPU"
  "                   ","                             ","Use OpenCL for AMD GPU or other devices"
  "                   ","Cross-platform               ","To extend from Linux to MacOS"
  "                   ","                             ","Large image models, e.g., `VGG <https://arxiv.org/pdf/1409.1556.pdf>`_ and `Residual Net <http://arxiv.org/abs/1512.03385>`_"
  "v1.1 Jan 2017      ","Model Zoo                    ","GoogleNet; Health-care models"
  "                   ","Caffe converter              ","Use SINGA to train models configured in caffe proto files"
  "                   ","Model components             ","Add concat and slice layers; accept multiple inputs to the net"
  "                   ","Compilation and installation ","Windows suppport"
  "                   ","                             ","Simplify the installation by compiling protobuf and openblas together with SINGA"
  "                   ","                             ","Build python wheel automatically using Jenkins"
  "                   ","                             ","Install SINGA from Debian packages"
  "v1.2 April 2017    ","Numpy API                    ","Implement functions for the tensor module of PySINGA following numpy API"
  "                   ","Distributed training         ","Migrate distributed training frameworks from V0.3"
  "v1.3 July 2017     ","Memory optimization          ","Replace CNMEM with new memory pool to reduce memory footprint"
  "                   ","Execution optimization       ","Runtime optimization of execution scheduling"
