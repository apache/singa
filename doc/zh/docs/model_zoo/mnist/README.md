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
# 在MNIST数据集上训练RBM模型

这个例子使用MNIST数据集来训练一个RBM模型。RBM模型及其超参数参考[Hinton的论文](http://www.cs.toronto.edu/~hinton/science.pdf)中的设定。


## 操作说明

* 下载预处理的[MNIST数据集](https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz)。

* 开始训练，

        python train.py mnist.pkl.gz

  	默认情况下，训练代码将在CPU上运行。 要在GPU卡上运行它，请使用附加参数启动程序，

        python train.py mnist.pkl.gz --use_gpu
