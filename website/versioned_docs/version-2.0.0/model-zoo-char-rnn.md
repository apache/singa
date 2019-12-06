---
id: version-2.0.0-model-zoo-char-rnn
title: Train Char-RNN over plain text
original_id: model-zoo-char-rnn
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

Recurrent neural networks (RNN) are widely used for modelling sequential data, e.g., natural language sentences. This example describes how to implement a RNN application (or model) using SINGA's RNN layers. We will use the [char-rnn](https://github.com/karpathy/char-rnn) model as an example, which trains over sentences or source code, with each character as an input unit. Particularly, we will train a RNN using GRU over Linux kernel source code. After training, we expect to generate meaningful code from the model.

## Instructions

### Compile and install SINGA.

Currently the RNN implementation depends on Cudnn with version >= 5.05.

### Prepare the dataset.

Download the [kernel source code](http://cs.stanford.edu/people/karpathy/char-rnn/). Other plain text files can also be used.

### Start the training,

> Please `cd` to `singa/examples/char-rnn/` for the following commands

```shell
python train.py linux_input.txt
```

Some hyper-parameters could be set through command line,

```shell
python train.py -h
```

### Sample characters

Sample characters from the model by providing the number of characters to sample and the seed string.

```shell
python sample.py 'model.bin' 100 --seed '#include <std'
```

Please replace 'model.bin' with the path to one of the checkpoint paths.
