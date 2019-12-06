---
id: version-2.0.0-RELEASE_NOTES_2.0.0
title: singa-incubating-2.0.0 Release Notes
original_id: RELEASE_NOTES_2.0.0
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

SINGA is a general distributed deep learning platform for training big deep learning models over large datasets.

This release includes following features:

- Core components

  - [SINGA-434] Support tensor broadcasting
  - [SINGA-370] Improvement to tensor reshape and various misc. changes related to SINGA-341 and 351

- Model components

  - [SINGA-333] Add support for Open Neural Network Exchange (ONNX) format
  - [SINGA-385] Add new python module for optimizers
  - [SINGA-394] Improve the CPP operations via Intel MKL DNN lib
  - [SINGA-425] Add 3 operators , Abs(), Exp() and leakyrelu(), for Autograd
  - [SINGA-410] Add two function, set_params() and get_params(), for Autograd Layer class
  - [SINGA-383] Add Separable Convolution for autograd
  - [SINGA-388] Develop some RNN layers by calling tiny operations like matmul, addbias.
  - [SINGA-382] Implement concat operation for autograd
  - [SINGA-378] Implement maxpooling operation and its related functions for autograd
  - [SINGA-379] Implement batchnorm operation and its related functions for autograd

- Utility functions and CI

  - [SINGA-432] Update depdent lib versions in conda-build config
  - [SINGA-429] Update docker images for latest cuda and cudnn
  - [SINGA-428] Move Docker images under Apache user name

- Documentation and usability
  - [SINGA-395] Add documentation for autograd APIs
  - [SINGA-344] Add a GAN example
  - [SINGA-390] Update installation.md
  - [SINGA-384] Implement ResNet using autograd API
  - [SINGA-352] Complete SINGA documentation in Chinese version

* Bugs fixed
  - [SINGA-431] Unit Test failed - Tensor Transpose
  - [SINGA-422] ModuleNotFoundError: No module named "\_singa_wrap"
  - [SINGA-418] Unsupportive type 'long' in python3.
  - [SINGA-409] Basic `singa-cpu` import throws error
  - [SINGA-408] Unsupportive function definition in python3
  - [SINGA-380] Fix bugs from Reshape
