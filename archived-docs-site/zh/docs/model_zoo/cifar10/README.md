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
# 在Cifar-10上训练CNN

卷积神经网络（CNN）是一种被广泛用于图像和视频分类的前馈人造神经网络。在此例子中，我们将在Cifar-10数据集上训练三个深度CNN模型来进行图像分类，

1. [AlexNet](https://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layers-18pct.cfg)，我们在验证集上能达到的最高准确度（不做数据增强）在82%左右。
2. [VGGNet](http://torch.ch/blog/2015/07/30/cifar.html)，我们在验证集上能达到的最高准确度（不做数据增强）在89%左右。
3. [ResNet](https://github.com/facebook/fb.resnet.torch)，我们在验证集上能达到的最高准确度（不做数据增强）在83%左右。
4. [来自Caffe的AlexNet](https://github.com/BVLC/caffe/tree/master/examples/cifar10)，SINGA能够无缝转换Caffe模型。


## 操作说明


### SINGA安装

用户可以从源码编译和安装C++或者Python版本的SINGA。代码可以在CPU和GPU上执行。对于GPU上做训练，CUDA和CUDNN（V4或V5）是需要的。请参考安装界面以获得详细指示。

### 数据准备

Cifar-10的二进制数据集文件可以由如下指令下载

        python download_data.py bin

Python版本可以由如下指令下载

        python download_data.py py

### 训练

这里有4个训练程序

1.train.py。下面的指令将会用'cifar-10-batches-py'底下的python版本的Cifar-10数据集训练VGG模型。

        python train.py vgg cifar-10-batches-py

  要训练其他模型，请用'alexnet'，'resnet'或'caffe'替换'vgg'。其中，'caffe'指由caffe的AlexNet转换的模型。默认情况下，模型将在CudaGPU设备上训练；若想要在CppCPU上运行，需添加额外的参数

        python train.py vgg cifar-10-batches-py  --use_cpu

2.alexnet.cc。它通过调用CPP API在CudaGPU上训练AlexNet模型。

        ./run.sh

3.alexnet-parallel.cc。它通过调用CPP API在两个CudaGPU上训练AlexNet模型。两个设备上同时运行并计算模型参数的梯度，然后它们会在CPU设备上取平均并用于更新参数。

        ./run-parallel.sh

4.vgg-parallel.cc。它调用CPP API在两个CudaGPU设备上训练VGG模型，同alexnet-parallel.cc类似。

### 预测

predict.py包含预测函数

        def predict(net, images, dev, topk=5)

net通过加载先前训练好的模型来被创建；images包含图像的numpy数组（每张图像一行）；dev是训练时的设备，例如，一个CudaGPU设备或者CppCPU设备；它将返回每个实例的topk标签。

Predict.py文件主函数提供了用预训练的模型为新图片做预测的例子。'mode.bin'文件由训练程序生成，需要被放置在cifar目录下执行

        python predict.py
