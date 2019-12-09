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
name: GoogleNet on ImageNet

SINGA version: 1.0.1

SINGA commit: 8c990f7da2de220e8a012c6a8ecc897dc7532744

parameter_url: https://s3-ap-southeast-1.amazonaws.com/dlfile/bvlc_googlenet.tar.gz

parameter_sha1: 0a88e8948b1abca3badfd8d090d6be03f8d7655d

license: unrestricted https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet

---

# 用GoogleNet做图像分类


这个例子中，我们将caffe训练好的GoogleNet转换为SINGA模型以用作图像分类。

## 操作说明

* 下载参数的checkpoint文件到如下目录

        $ wget https://s3-ap-southeast-1.amazonaws.com/dlfile/bvlc_googlenet.tar.gz
        $ tar xvf bvlc_googlenet.tar.gz

* 运行程序

        # use cpu
        $ python serve.py -C &
        # use gpu
        $ python serve.py &

* 提交图片进行分类

        $ curl -i -F image=@image1.jpg http://localhost:9999/api
        $ curl -i -F image=@image2.jpg http://localhost:9999/api
        $ curl -i -F image=@image3.jpg http://localhost:9999/api

image1.jpg, image2.jpg和image3.jpg应该在执行指令前就已被下载。

## 详细信息

我们首先从[Caffe的checkpoint文件](http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel)中提取参数值，并转换为pickle版本。下载checkpoint文件后进入`caffe_root/python`文件夹，运行如下脚本：

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

然后我们使用SINGA的FeedForwardNet结构构建GoogleNet。 请注意，我们添加了一个EndPadding层来解决Caffe（下取整）和cuDNN（上取整）之间池化图层舍入策略差异的问题。 只有MaxPooling图层以外的启动块才有此问题。 参考[这里](http://joelouismarino.github.io/blog_posts/blog_googlenet_keras.html)更多详情。
