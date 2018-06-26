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
# SINGA Docker Images

## Availabe images


| Tag | OS version | devel/runtime | Device|CUDA/CUDNN|Python|
|:----|:-----------|:--------------|:------|:---------|:-----|
|runtime| Ubuntu16.04|runtime|CPU|-|3.6|
|conda-cuda9.0| Ubuntu16.04|devel|GPU|CUDA9.0+CUDNN7.1.2|3.6|
|cuda9.0-py2| Ubuntu16.04|devel|GPU|CUDA9.0+CUDNN7.1.2|2.7|
|cuda9.0-py3| Ubuntu16.04|devel|GPU|CUDA9.0+CUDNN7.1.2|3.6|

runtime and conda-xxx image has installed miniconda3;
cudaxxx images have installed all depedent libs using apt-get.

## Usage

    docker pull nusdbsystem/singa:<Tag>
    docker run -it nusdbsystem/singa:<Tag> /bin/bash
    nvidia-docker run -it nusdbsystem/singa:<Tag> /bin/bash
