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
name: Resnets on ImageNet SINGA version: 1.1 SINGA commit: 45ec92d8ffc1fa1385a9307fdf07e21da939ee2f parameter_url: https://s3-ap-southeast-1.amazonaws.com/dlfile/resnet/resnet-18.tar.gz license: Apache V2, https://github.com/facebook/fb.resnet.torch/blob/master/LICENSE

---

# 用ResNet做图像分类


这个例子中，我们将Torch训练好的ResNet转换为SINGA模型以用作图像分类。

## 操作说明

* 下载参数的checkpoint文件到如下目录

        $ wget https://s3-ap-southeast-1.amazonaws.com/dlfile/resnet/resnet-18.tar.gz
		$ wget https://s3-ap-southeast-1.amazonaws.com/dlfile/resnet/synset_words.txt
		$ tar xvf resnet-18.tar.gz

* 运行程序

        $ python serve.py -h

* 运行程序

        # use cpu
		$ python serve.py --use_cpu --parameter_file resnet-18.pickle --model resnet --depth 18 &
  		# use gpu
		$ python serve.py --parameter_file resnet-18.pickle --model resnet --depth 18 &

	我们提供了以下模型和深度配置的参数文件:
	* resnet (原始 resnet), [18](https://s3-ap-southeast-1.amazonaws.com/dlfile/resnet/resnet-18.tar.gz)|[34](https://s3-ap-southeast-1.amazonaws.com/dlfile/resnet/resnet-34.tar.gz)|[101](https://s3-ap-southeast-1.amazonaws.com/dlfile/resnet/resnet-101.tar.gz)|[152](https://s3-ap-southeast-1.amazonaws.com/dlfile/resnet/resnet-152.tar.gz)
  	* 包括批量正则, [50](https://s3-ap-southeast-1.amazonaws.com/dlfile/resnet/resnet-50.tar.gz)
  	* wrn (宽 resnet), [50](https://s3-ap-southeast-1.amazonaws.com/dlfile/resnet/wrn-50-2.tar.gz)
  	* preact (包括 pre-activation 的 resnet) [200](https://s3-ap-southeast-1.amazonaws.com/dlfile/resnet/resnet-200.tar.gz)


* 提交图片进行分类

        $ curl -i -F image=@image1.jpg http://localhost:9999/api
        $ curl -i -F image=@image2.jpg http://localhost:9999/api
        $ curl -i -F image=@image3.jpg http://localhost:9999/api

image1.jpg, image2.jpg和image3.jpg应该在执行指令前就已被下载。

## 详细信息

用`convert.py`从torch参数文件中提取参数值

* 运行程序

    	$ python convert.py -h
