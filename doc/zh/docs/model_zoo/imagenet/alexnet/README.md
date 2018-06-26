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
# 在ImageNet上训练AlexNet

卷积神经网络（CNN）是一种广泛用于图像和视频分类的前馈神经网络。 在这个例子中，我们将使用[深度CNN模型](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)来对ImageNet数据集进行图像分类。

## 操作说明

### 编译SINGA

请用CUDA，CUDNN和OpenCV编译SINGA。 您可以手动打开CMakeLists.txt中的选项或在build /文件夹中运行`ccmake ..`进行配置。

我们已经在CuDNN V4和V5（V5需要Cuda7.5）上进行了测试。


### 数据下载
* 请参考创建[ImageNet 2012数据集](https://github.com/amd/OpenCL-caffe/wiki/Instructions-to-create-ImageNet-2012-data)的步骤1-3下载和加压数据。
* 你可以通过[get_ilsvrc_aux.sh](https://github.com/BVLC/caffe/blob/master/data/ilsvrc12/get_ilsvrc_aux.sh)或从[ImageNet](http://www.image-net.org/download-images)下载训练和验证集。

### 数据预处理
* 假设你已经下载了数据和描述文件。 现在我们应该将数据转换为二进制文件。你可以运行：

          sh create_data.sh

  这个脚本会在指定输出目录下生成一个测试文件（`test.bin`），均值文件（`mean.bin`）和一些训练文件（`trainX.bin`）。
* 你也可以改变`create_data.sh`的参数。
  + `-trainlist <file>`: 训练数据列表文件;
  + `-trainfolder <folder>`: 训练图片所在文件夹;
  + `-testlist <file>`: 测试数据列表文件;
  + `-testfolder <floder>`: 测试图像所在文件夹;
  + `-outdata <folder>`: 保存输出文件的文件夹，包括平均值，训练和测试文件。 该脚本将在指定的文件夹中生成这些文件;
  + `-filesize <int>`: 在每个二进制文件中存储的训练图片个数.

### 训练
* 准备好数据后，你可以运行下面指令来训练AlexNet模型。

          sh run.sh

* 你可以改变`run.sh`的参数。
  + `-epoch <int>`: 要训练的epoch数目，默认为90;
  + `-lr <float>`: 基础学习率，学习率将减少每20个时期，更具体地，lr = lr * exp（0.1 *（epoch / 20））;
  + `-batchsize <int>`: 批数目，它应该根据你的内存而改变;
  + `-filesize <int>`: 存储在每个二进制文件中的训练图像的数量，与数据预处理中的文件大小相同;
  + `-ntrain <int>`: 训练图片的数目;
  + `-ntest <int>`: 测试图片的数目;
  + `-data <folder>`: 存储二进制文件的文件夹，它恰好是数据预处理步骤中的输出文件夹;
  + `-pfreq <int>`: 打印当前模型状态（损失和准确度）的频率（以批数据为单位）;
  + `-nthreads <int>`: 加载传给模型的数据所有的线程数。
