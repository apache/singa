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
name: VGG models on ImageNet
SINGA version: 1.1.1
SINGA commit:
license: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
---

# Image Classification using VGG


In this example, we convert VGG on [PyTorch](https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py)
to SINGA for image classification.

## Instructions

* Download one parameter checkpoint file (see below) and the synset word file of ImageNet into this folder, e.g.,

        $ wget https://s3-ap-southeast-1.amazonaws.com/dlfile/vgg/vgg11.tar.gz
        $ wget https://s3-ap-southeast-1.amazonaws.com/dlfile/resnet/synset_words.txt
        $ tar xvf vgg11.tar.gz

* Usage

        $ python serve.py -h

* Example

        # use cpu
        $ python serve.py --use_cpu --parameter_file vgg11.pickle --depth 11 &
        # use gpu
        $ python serve.py --parameter_file vgg11.pickle --depth 11 &

  The parameter files for the following model and depth configuration pairs are provided:
  * Without batch-normalization, [11](https://s3-ap-southeast-1.amazonaws.com/dlfile/vgg/vgg11.tar.gz), [13](https://s3-ap-southeast-1.amazonaws.com/dlfile/vgg/vgg13.tar.gz), [16](https://s3-ap-southeast-1.amazonaws.com/dlfile/vgg/vgg16.tar.gz), [19](https://s3-ap-southeast-1.amazonaws.com/dlfile/vgg/vgg19.tar.gz)
  * With batch-normalization, [11](https://s3-ap-southeast-1.amazonaws.com/dlfile/vgg/vgg11_bn.tar.gz), [13](https://s3-ap-southeast-1.amazonaws.com/dlfile/vgg/vgg13_bn.tar.gz), [16](https://s3-ap-southeast-1.amazonaws.com/dlfile/vgg/vgg16_bn.tar.gz), [19](https://s3-ap-southeast-1.amazonaws.com/dlfile/vgg/vgg19_bn.tar.gz)

* Submit images for classification

        $ curl -i -F image=@image1.jpg http://localhost:9999/api
        $ curl -i -F image=@image2.jpg http://localhost:9999/api
        $ curl -i -F image=@image3.jpg http://localhost:9999/api

image1.jpg, image2.jpg and image3.jpg should be downloaded before executing the above commands.

## Details

The parameter files were converted from the pytorch via the convert.py program.

Usage:

    $ python convert.py -h
