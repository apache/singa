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
name: VGG on ImageNet SINGA version: 1.1.1 SINGA commit: license: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py

---

# 用VGG做图像分类


这个例子中，我们将PyTorch训练好的VGG转换为SINGA模型以用作图像分类。

## 操作说明

* 下载参数的checkpoint文件到如下目录

        $ wget https://s3-ap-southeast-1.amazonaws.com/dlfile/vgg/vgg11.tar.gz
		$ wget https://s3-ap-southeast-1.amazonaws.com/dlfile/resnet/synset_words.txt
		$ tar xvf vgg11.tar.gz

* 运行程序

        $ python serve.py -h

* 例子

        # use cpu
		$ python serve.py --use_cpu --parameter_file vgg11.pickle --depth 11 &
  		# use gpu
		$ python serve.py --parameter_file vgg11.pickle --depth 11 &

	我们提供了以下模型和深度配置的参数文件:
	* 不使用批量正则, [11](https://s3-ap-southeast-1.amazonaws.com/dlfile/vgg/vgg11.tar.gz), [13](https://s3-ap-southeast-1.amazonaws.com/dlfile/vgg/vgg13.tar.gz), [16](https://s3-ap-southeast-1.amazonaws.com/dlfile/vgg/vgg16.tar.gz), [19](https://s3-ap-southeast-1.amazonaws.com/dlfile/vgg/vgg19.tar.gz)
	* 使用批量正则, [11](https://s3-ap-southeast-1.amazonaws.com/dlfile/vgg/vgg11_bn.tar.gz), [13](https://s3-ap-southeast-1.amazonaws.com/dlfile/vgg/vgg13_bn.tar.gz), [16](https://s3-ap-southeast-1.amazonaws.com/dlfile/vgg/vgg16_bn.tar.gz), [19](https://s3-ap-southeast-1.amazonaws.com/dlfile/vgg/vgg19_bn.tar.gz)


* 提交图片进行分类

        $ curl -i -F image=@image1.jpg http://localhost:9999/api
        $ curl -i -F image=@image2.jpg http://localhost:9999/api
        $ curl -i -F image=@image3.jpg http://localhost:9999/api

image1.jpg, image2.jpg和image3.jpg应该在执行指令前就已被下载。

## 详细信息

用`convert.py`从Pytorch参数文件中提取参数值

* 运行程序

    	$ python convert.py -h
