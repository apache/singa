---
id: version-2.0.0-RELEASE_NOTES_0.3.0
title: singa-incubating-0.3.0 Release Notes
original_id: RELEASE_NOTES_0.3.0
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

SINGA is a general distributed deep learning platform for training big deep learning models over large datasets. It is designed with an intuitive programming model based on the layer abstraction. SINGA supports a wide variety of popular deep learning models.

This release includes following features:

- GPU Support

  - [SINGA-131] Implement and optimize hybrid training using both CPU and GPU
  - [SINGA-136] Support cuDNN v4
  - [SINGA-134] Extend SINGA to run over a GPU cluster
  - [SINGA-157] Change the priority of cudnn library and install libsingagpu.so

- Remove Dependences

  - [SINGA-156] Remove the dependency on ZMQ for single process training
  - [SINGA-155] Remove zookeeper for single-process training

- Python Binding

  - [SINGA-126] Python Binding for Interactive Training

- Other Improvements

  - [SINGA-80] New Blob Level and Address Level Math Operation Interface
  - [SINGA-130] Data Prefetching
  - [SINGA-145] New SGD based optimization Updaters: AdaDelta, Adam, AdamMax

- Bugs Fixed
  - [SINGA-148] Race condition between Worker threads and Driver
  - [SINGA-150] Mesos Docker container failed
  - [SIGNA-141] Undesired Hash collision when locating process id to worker…
  - [SINGA-149] Docker build fail
  - [SINGA-143] The compilation cannot detect libsingagpu.so file
