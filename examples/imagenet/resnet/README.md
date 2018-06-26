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
---
name: Resnets on ImageNet
SINGA version: 1.1
SINGA commit: 45ec92d8ffc1fa1385a9307fdf07e21da939ee2f
parameter_url: https://s3-ap-southeast-1.amazonaws.com/dlfile/resnet/resnet-18.tar.gz
license: Apache V2, https://github.com/facebook/fb.resnet.torch/blob/master/LICENSE
---

# Image Classification using Residual Networks


In this example, we convert Residual Networks trained on [Torch](https://github.com/facebook/fb.resnet.torch) to SINGA for image classification.

## Instructions

* Download one parameter checkpoint file (see below) and the synset word file of ImageNet into this folder, e.g.,

        $ wget https://s3-ap-southeast-1.amazonaws.com/dlfile/resnet/resnet-18.tar.gz
        $ wget https://s3-ap-southeast-1.amazonaws.com/dlfile/resnet/synset_words.txt
        $ tar xvf resnet-18.tar.gz

* Usage

        $ python serve.py -h

* Example

        # use cpu
        $ python serve.py --use_cpu --parameter_file resnet-18.pickle --model resnet --depth 18 &
        # use gpu
        $ python serve.py --parameter_file resnet-18.pickle --model resnet --depth 18 &

  The parameter files for the following model and depth configuration pairs are provided:
  * resnet (original resnet), [18](https://s3-ap-southeast-1.amazonaws.com/dlfile/resnet/resnet-18.tar.gz)|[34](https://s3-ap-southeast-1.amazonaws.com/dlfile/resnet/resnet-34.tar.gz)|[101](https://s3-ap-southeast-1.amazonaws.com/dlfile/resnet/resnet-101.tar.gz)|[152](https://s3-ap-southeast-1.amazonaws.com/dlfile/resnet/resnet-152.tar.gz)
  * addbn (resnet with a batch normalization layer after the addition), [50](https://s3-ap-southeast-1.amazonaws.com/dlfile/resnet/resnet-50.tar.gz)
  * wrn (wide resnet), [50](https://s3-ap-southeast-1.amazonaws.com/dlfile/resnet/wrn-50-2.tar.gz)
  * preact (resnet with pre-activation) [200](https://s3-ap-southeast-1.amazonaws.com/dlfile/resnet/resnet-200.tar.gz)

* Submit images for classification

        $ curl -i -F image=@image1.jpg http://localhost:9999/api
        $ curl -i -F image=@image2.jpg http://localhost:9999/api
        $ curl -i -F image=@image3.jpg http://localhost:9999/api

image1.jpg, image2.jpg and image3.jpg should be downloaded before executing the above commands.

## Details

The parameter files were extracted from the original [torch files](https://github.com/facebook/fb.resnet.torch/tree/master/pretrained) via
the convert.py program.

Usage:

    $ python convert.py -h
