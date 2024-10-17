<!--
    Licensed to the Apache Software Foundation (ASF) under one
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
-->

# Image Classification using Convolutional Neural Networks

Examples inside this folder show how to train CNN models using
SINGA for image classification.

* `data` includes the scripts for preprocessing image datasets.
  Currently, MNIST, CIFAR10 and CIFAR100 are included.

* `model` includes the CNN model construction codes by creating
  a subclass of `Module` to wrap the neural network operations
  of each model. Then computational graph is enabled to optimized
  the memory and efficiency.

* `autograd` includes the codes to train CNN models by calling the
  [neural network operations](../../python/singa/autograd.py) imperatively.
  The computational graph is not created.

* `train_cnn.py` is the training script, which controls the training flow by
  doing BackPropagation and SGD update.

* `train_multiprocess.py` is the script for distributed training on a single
  node with multiple GPUs; it uses Python's multiprocessing module and NCCL.

* `train_mpi.py` is the script for distributed training (among multiple nodes)
  using MPI and NCCL for communication.

* `benchmark.py` tests the training throughput using `ResNet50` as the workload.
