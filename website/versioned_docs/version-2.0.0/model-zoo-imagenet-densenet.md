---
id: version-2.0.0-model-zoo-imagenet-densenet
title: Image Classification using DenseNet
original_id: model-zoo-imagenet-densenet
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

In this example, we convert DenseNet on [PyTorch](https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py) to SINGA for image classification.

## Instructions

> Please `cd` to `singa/examples/imagenet/densenet/` for the following commands

### Download one parameter checkpoint file

Download one parameter checkpoint file (see below) and the synset word file of ImageNet into this folder, e.g.,

```shell
$ wget https://s3-ap-southeast-1.amazonaws.com/dlfile/densenet/densenet-121.tar.gz
$ wget https://s3-ap-southeast-1.amazonaws.com/dlfile/resnet/synset_words.txt
$ tar xvf densenet-121.tar.gz
```

### Usage

```shell
$ python serve.py -h
```

### Example

```shell
# use cpu
$ python serve.py --use_cpu --parameter_file densenet-121.pickle --depth 121 &
# use gpu
$ python serve.py --parameter_file densenet-121.pickle --depth 121 &
```

The parameter files for the following model and depth configuration pairs are provided: [121](https://s3-ap-southeast-1.amazonaws.com/dlfile/densenet/densenet-121.tar.gz), [169](https://s3-ap-southeast-1.amazonaws.com/dlfile/densenet/densenet-169.tar.gz), [201](https://s3-ap-southeast-1.amazonaws.com/dlfile/densenet/densenet-201.tar.gz), [161](https://s3-ap-southeast-1.amazonaws.com/dlfile/densenet/densenet-161.tar.gz)

### Submit

Submit images for classification

```shell
$ curl -i -F image=@image1.jpg http://localhost:9999/api
$ curl -i -F image=@image2.jpg http://localhost:9999/api
$ curl -i -F image=@image3.jpg http://localhost:9999/api
```

image1.jpg, image2.jpg and image3.jpg should be downloaded before executing the above commands.

## Details

The parameter files were converted from the pytorch via the convert.py program.

Usage:

```shell
$ python convert.py -h
```
