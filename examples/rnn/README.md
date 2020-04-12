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
# Train Char-RNN over plain text

Recurrent neural networks (RNN) are widely used for modelling sequential data,
e.g., natural language sentences. This example describes how to implement a RNN
application (or model) using SINGA's RNN layers.
We will use the [char-rnn](https://github.com/karpathy/char-rnn) model as an
example, which trains over sentences or
source code, with each character as an input unit. Particularly, we will train
a RNN over Linux kernel source code. 

## Instructions

* Prepare the dataset. Download the [kernel source code](http://cs.stanford.edu/people/karpathy/char-rnn/).
Other plain text files can also be used.

* Start the training,

        python train.py linux_input.txt

  Some hyper-parameters could be set through command line,

        python train.py -h
