---
id: version-2.0.0-model-zoo-rbm-mnist
title: Train a RBM model against MNIST dataset
original_id: model-zoo-rbm-mnist
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

This example is to train an RBM model using the MNIST dataset. The RBM model and its hyper-parameters are set following [Hinton's paper](http://www.cs.toronto.edu/~hinton/science.pdf)

## Running instructions

### Download

Download the pre-processed [MNIST dataset](https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz)

### Start the training

> Please `cd` to `singa/examples/mnist/` for the following commands

```shell
python train.py mnist.pkl.gz
```

By default the training code would run on CPU. To run it on a GPU card, please start the program with an additional argument

```shell
python train.py mnist.pkl.gz --use_gpu
```
