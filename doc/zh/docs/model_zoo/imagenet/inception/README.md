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
name: Inception V4 on ImageNet SINGA version: 1.1.1 SINGA commit: parameter_url: https://s3-ap-southeast-1.amazonaws.com/dlfile/inception_v4.tar.gz parameter_sha1: 5fdd6f5d8af8fd10e7321d9b38bb87ef14e80d56 license: https://github.com/tensorflow/models/tree/master/slim

---

# 用Inception V4做图像分类


这个例子中，我们将Tensorflow训练好的Inception V4转换为SINGA模型以用作图像分类。

## 操作说明

* 下载参数的checkpoint文件到如下目录

        $ wget
		$ tar xvf inception_v4.tar.gz

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

我们首先从Tensorflow的checkpoint文件中提取参数值到一个pickle版本中。 下载并解压缩checkpoint文件后，运行以下脚本

	$ python convert.py --file_name=inception_v4.ckpt
