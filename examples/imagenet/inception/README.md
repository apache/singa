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
name: Inception V4 on ImageNet
SINGA version: 1.1.1
SINGA commit:
parameter_url: https://s3-ap-southeast-1.amazonaws.com/dlfile/inception_v4.tar.gz
parameter_sha1: 5fdd6f5d8af8fd10e7321d9b38bb87ef14e80d56
license: https://github.com/tensorflow/models/tree/master/slim
---

# Image Classification using Inception V4

In this example, we convert Inception V4 trained on Tensorflow to SINGA for image classification.

## Instructions

* Download the parameter checkpoint file

        $ wget
        $ tar xvf inception_v4.tar.gz

* Download [synset_word.txt](https://github.com/BVLC/caffe/blob/master/data/ilsvrc12/get_ilsvrc_aux.sh) file.

* Run the program

        # use cpu
        $ python serve.py -C &
        # use gpu
        $ python serve.py &

* Submit images for classification

        $ curl -i -F image=@image1.jpg http://localhost:9999/api
        $ curl -i -F image=@image2.jpg http://localhost:9999/api
        $ curl -i -F image=@image3.jpg http://localhost:9999/api

image1.jpg, image2.jpg and image3.jpg should be downloaded before executing the above commands.

## Details

We first extract the parameter values from [Tensorflow's checkpoint file](http://download.tensorflow.org/models/inception_v4_2016_09_09.tar.gz) into a pickle version.
After downloading and decompressing the checkpoint file, run the following script

    $ python convert.py --file_name=inception_v4.ckpt
