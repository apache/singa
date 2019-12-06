---
id: version-2.0.0-RELEASE_NOTES_1.2.0
title: singa-incubating-1.2.0 Release Notes
original_id: RELEASE_NOTES_1.2.0
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

SINGA is a general distributed deep learning platform for training big deep learning models over large datasets.

This release includes following features:

- Core components

  - [SINGA-290] Upgrade to Python 3
  - [SINGA-341] Added stride functionality to tensors for CPP
  - [SINGA-347] Create a function that supports einsum
  - [SINGA-351] Added stride support and cudnn codes to cuda

- Model components

  - [SINGA-300] Add residual networks for imagenet classification
  - [SINGA-312] Rename layer parameters
  - [SINGA-313] Add L2 norm layer
  - [SINGA-315] Reduce memory footprint by Python generator for parameter
  - [SINGA-316] Add SigmoidCrossEntropy
  - [SINGA-324] Extend RNN layer to accept variant seq length across batches
  - [SINGA-326] Add Inception V4 for ImageNet classification
  - [SINGA-328] Add VGG models for ImageNet classification
  - [SINGA-329] Support layer freezing during training (fine-tuning)
  - [SINGA-346] Update cudnn from V5 to V7
  - [SINGA-349] Create layer operations for autograd
  - [SINGA-363] Add DenseNet for Imagenet classification

- Utility functions and CI

  - [SINGA-274] Improve Debian packaging with CPack
  - [SINGA-303] Create conda packages
  - [SINGA-337] Add test cases for code
  - [SINGA-348] Support autograd MLP Example
  - [SINGA-345] Update Jenkins and fix bugs in compliation
  - [SINGA-354] Update travis scripts to use conda-build for all platforms
  - [SINGA-358] Consolidated RUN steps and cleaned caches in Docker containers
  - [SINGA-359] Create alias for conda packages

- Documentation and usability

  - [SINGA-223] Fix side navigation menu in the website
  - [SINGA-294] Add instructions to run CUDA unit tests on Windows
  - [SINGA-305] Add jupyter notebooks for SINGA V1 tutorial
  - [SINGA-319] Fix link errors on the index page
  - [SINGA-352] Complete SINGA documentation in Chinese version
  - [SINGA-361] Add git instructions for contributors and committers

- Bugs fixed
  - [SINGA-330] fix openblas building on i7 7700k
  - [SINGA-331] Fix the bug of tensor division operation
  - [SINGA-350] Error from python3 test
  - [SINGA-356] Error using travis tool to build SINGA on mac os
  - [SINGA-363] Fix some bugs in imagenet examples
  - [SINGA-368] Fix the bug in Cifar10 examples
  - [SINGA-369] the errors of examples in testing
