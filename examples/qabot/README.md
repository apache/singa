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
# Train a question and answering QABOT model

This example describes how to implement a question and answering QABOT
application using SINGA's CUDNN RNN layers.

We will use the [LSTM](https://www.mitpressjournals.org/doi/abs/10.1162/neco.1997.9.8.1735) model together with max pooling as an
example to train the QABOT.

## Instructions

* Start the training,

        python qabot_train.py
