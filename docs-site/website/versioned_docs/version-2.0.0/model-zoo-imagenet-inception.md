---
id: version-2.0.0-model-zoo-imagenet-inception
title: Image Classification using Inception V4
original_id: model-zoo-imagenet-inception
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

In this example, we convert Inception V4 trained on Tensorflow to SINGA for image classification. Tested on SINGA version 1.1.1 with [parameters pretrained by tensorflow](https://s3-ap-southeast-1.amazonaws.com/dlfile/inception_v4.tar.gz).

## Instructions

> Please `cd` to `singa/examples/imagenet/inception/` for the following commands

### Download

Download the parameter checkpoint file

```shell
$ wget https://s3-ap-southeast-1.amazonaws.com/dlfile/inception_v4.tar.gz
$ tar xvf inception_v4.tar.gz
```

Download [synset_word.txt](https://github.com/BVLC/caffe/blob/master/data/ilsvrc12/get_ilsvrc_aux.sh) file.

### Run the program

```shell
# use cpu
$ python serve.py -C &
# use gpu
$ python serve.py &
```

### Submit images for classification

```shell
$ curl -i -F image=@image1.jpg http://localhost:9999/api
$ curl -i -F image=@image2.jpg http://localhost:9999/api
$ curl -i -F image=@image3.jpg http://localhost:9999/api
```

image1.jpg, image2.jpg and image3.jpg should be downloaded before executing the above commands.

## Details

We first extract the parameter values from [Tensorflow's checkpoint file](http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz) into a pickle version. After downloading and decompressing the checkpoint file, run the following script

```shell
$ python convert.py --file_name=inception_v4.ckpt
```
