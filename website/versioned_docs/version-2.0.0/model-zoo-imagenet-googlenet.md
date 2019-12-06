---
id: version-2.0.0-model-zoo-imagenet-googlenet
title: Image Classification using GoogleNet
original_id: model-zoo-imagenet-googlenet
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

In this example, we convert GoogleNet trained on Caffe to SINGA for image classification. Tested on [SINGA commit](https://github.com/apache/singa/commit/8c990f7da2de220e8a012c6a8ecc897dc7532744) with [the parameters](https://s3-ap-southeast-1.amazonaws.com/dlfile/bvlc_googlenet.tar.gz).

## Instructions

> Please `cd` to `singa/examples/imagenet/googlenet/` for the following commands

### Download

Download the parameter checkpoint file into this folder

```shell
$ wget https://s3-ap-southeast-1.amazonaws.com/dlfile/bvlc_googlenet.tar.gz
$ tar xvf bvlc_googlenet.tar.gz
```

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

We first extract the parameter values from [Caffe's checkpoint file](http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel) into a pickle version After downloading the checkpoint file into `caffe_root/python` folder, run the following script

```python
# to be executed within caffe_root/python folder
import caffe
import numpy as np
import cPickle as pickle

model_def = '../models/bvlc_googlenet/deploy.prototxt'
weight = 'bvlc_googlenet.caffemodel'  # must be downloaded at first
net = caffe.Net(model_def, weight, caffe.TEST)

params = {}
for layer_name in net.params.keys():
    weights=np.copy(net.params[layer_name][0].data)
    bias=np.copy(net.params[layer_name][1].data)
    params[layer_name+'_weight']=weights
    params[layer_name+'_bias']=bias
    print layer_name, weights.shape, bias.shape

with open('bvlc_googlenet.pickle', 'wb') as fd:
    pickle.dump(params, fd)
```

Then we construct the GoogleNet using SINGA's FeedForwardNet structure. Note that we added a EndPadding layer to resolve the issue from discrepancy of the rounding strategy of the pooling layer between Caffe (ceil) and cuDNN (floor). Only the MaxPooling layers outside inception blocks have this problem. Refer to [this](https://gist.github.com/joelouismarino/a2ede9ab3928f999575423b9887abd14) for more detials.
