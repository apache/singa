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
# 初始化器(Initializer)

## Python API

普遍使用的参数初始化方法（tensor对象）。

示例用法：

```python
from singa import tensor
from singa import initializer

x = tensor.Tensor((3, 5))
initializer.uniform(x, 3, 5) # use both fan_in and fan_out
initializer.uniform(x, 3, 0)  # use only fan_in
```
---

#### singa.initializer.uniform(t, fan_in=0, fan_out=0)

按照指定均匀分布对输入tensor初始化。

**参数：**
- **fan_in (int)** – 对于卷积层权重tensor，fan_in = nb_channel * kh * kw;对于全连接层，fan_in = input_feature_length
- **fan_out (int)** – 对于卷积层权重tensor，fan_out = nb_filter * kh * kw;对于全连接层，fan_out = output_feature_length

**参考文献** [Bengio and Glorot 2010]： Understanding the difficulty of training deep feedforward neuralnetworks.

---

#### singa.initializer.gaussian(t, fan_in=0, fan_out=0)

按照指定高斯分布对输入tensor初始化。

**参数：**
- **fan_in (int)** – 对于卷积层权重tensor，fan_in = nb_channel * kh * kw;对于全连接层，fan_in = input_feature_length
- **fan_out (int)** – 对于卷积层权重tensor，fan_out = nb_filter * kh * kw;对于全连接层，fan_out = output_feature_length

**参考文献** Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun: Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification

---
