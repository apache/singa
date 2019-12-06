---
id: version-2.0.0-benchmark-train
title: Benchmark for Distributed Training
original_id: benchmark-train
---

<!--- Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.  See the NOTICE file distributed with this work for additional information regarding copyright ownership.  The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.  You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the License for the specific language governing permissions and limitations under the License.  -->

Workload: we use a deep convolutional neural network, [ResNet-50](https://github.com/apache/singa/blob/master/examples/autograd/resnet.py) as the application. ResNet-50 is has 50 convolution layers for image classification. It requires 3.8 GFLOPs to pass a single image (of size 224x224) through the network. The input image size is 224x224.

Hardware: we use p2.8xlarge instances from AWS, each of which has 8 Nvidia Tesla K80 GPUs, 96 GB GPU memory in total, 32 vCPU, 488 GB main memory, 10 Gbps network bandwidth.

Metric: we measure the time per iteration for different number of workers to evaluate the scalability of SINGA. The batch size is fixed to be 32 per GPU. Synchronous training scheme is applied. As a result, the effective batch size is $32N$, where N is the number of GPUs. We compare with a popular open source system which uses the parameter server topology. The first GPU is selected as the server.

![Benchmark Experiments](assets/benchmark.png) <br/> **Scalability test. Bars are for the throughput; lines are for the communication cost.**
