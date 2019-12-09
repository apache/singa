.. Licensed to the Apache Software Foundation (ASF) under one
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


开发时间表
====================

.. csv-table::
  :header: "版本","模块","特性"

  "v0.1 2015 九月     ","神经网络                     ","前向传播神经网络, 包括 CNN, MLP"
  "                   ","                             ","类 RBM 模型, 包括 RBM"
  "                   ","                             ","循环神经网络, 包括标准 RNN"
  "                   ","架构                         ","在单节点运行一个工作组 (包括划分)"
  "                   ","                             ","在单节点运行多个工作组, 用 `Hogwild <http://www.eecs.berkeley.edu/~brecht/papers/hogwildTR.pdf>`_     "
  "                   ","                             ","分布式 Hogwild"
  "                   ","                             ","跨多节点运行多个工作组 , 如 `Downpour <http://papers.nips.cc/paper/4687-large-scale-distritbuted-deep-networks>`_"
  "                   ","                             ","All-Reduce 训练架构如 `DeepImage <http://arxiv.org/abs/1501.02876>`_                                "
  "                   ","                             ","服务器间负载均衡"
  "                   ","失败恢复                     ","检查点和恢复"
  "                   ","工具                         ","用 GNU 自动工具安装"
  "v0.2 2016 一月     ","神经网络                     ","前向传播神经网络, 包括 AlexNet, cuDNN 层, 工具"
  "                   ","                             ","循环神经网络, 包括 GRU 层和 BPTT"
  "                   ","                             ","模型划分和混合划分"
  "                   ","工具                         ","融合 Mesos 资源管理"
  "                   ","                             ","准备部署 Docker images"
  "                   ","                             ","可视化神经网络和调试信息"
  "                   ","绑定                         ","主要组件 Python 绑定"
  "                   ","GPU                          ","单节点多个 GPU "
  "v0.3 2016 四月     ","GPU                          ","多个节点, 每个包含多个 GPU"
  "                   ","                             ","用 GPU 和 CPU 混合训练  `CcT <http://arxiv.org/abs/1504.04343>`_"
  "                   ","                             ","支持 cuDNN v4 "
  "                   ","安装                         ","删除 ZeroMQ, CZMQ 依赖, 单节点训练 zookeeper"
  "                   ","优化器                       ","添加新的 SGD 优化器，包括 Adam, AdamMax 和 AdaDelta"
  "                   ","绑定                         ","增强 Python 绑定训练"
  "v1.0 2016 九月     ","模型抽象                     ","Tensor 基于线性代数, 神经网络和随机运算"
  "                   ","                             ","分布式参数更新优化器"
  "                   ","硬件                         ","使用 Cuda 和 Cudnn for Nvidia GPU"
  "                   ","                             ","使用 OpenCL for AMD GPU 及其他设备"
  "                   ","跨平台                       ","从 Linux 扩展到 MacOS"
  "                   ","                             ","大型图像模型, 例如, `VGG <https://arxiv.org/pdf/1409.1556.pdf>`_ 和 `Residual Net <http://arxiv.org/abs/1512.03385>`_"
  "v1.1 2017 一月     ","模型库                       ","GoogleNet; 医疗健康模型"
  "                   ","Caffe 转换器                 ","使用 SINGA 训练模型, 从 caffe proto 文件配置"
  "                   ","模型组件                     ","添加 concat 和 slice 层; 接受多个输入网络"
  "                   ","编译和安装                   ","Windows 支持"
  "                   ","                             ","通过与 protobuf 和 openblas 一起编译 SINGA 简化安装"
  "                   ","                             ","用 Jenkins 自动生成 python wheel"
  "                   ","                             ","从 Debian packages 安装 SINGA"
  "v1.2 2018 六月     ","AutoGrad                     ","后向传播 AutoGrad"
  "                   ","Python 3                     ","PySinga 支持 Python 3"
  "                   ","模型                         ","添加流行模型, 包括 VGG, ResNet, DenseNet, InceptionNet"
