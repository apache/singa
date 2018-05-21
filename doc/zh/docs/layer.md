# 层(Layer)

## Python API

Python层将C++层封装成更简单的API。

示例用法：

```python
from singa import layer
from singa import tensor
from singa import device

layer.engine = 'cudnn'  # to use cudnn layers
dev = device.create_cuda_gpu()

# create a convolution layer
conv = layer.Conv2D('conv', 32, 3, 1, pad=1, input_sample_shape=(3, 32, 32))
conv.to_device(dev)  # move the layer data onto a CudaGPU device
x = tensor.Tensor((3, 32, 32), dev)
x.uniform(-1, 1)
y = conv.foward(True, x)

dy = tensor.Tensor()
dy.reset_like(y)
dy.set_value(0.1)
# dp is a list of tensors for parameter gradients
dx, dp = conv.backward(kTrain, dy)
```

---

#### singa.layer.engine = 'cudnn'

引擎（engine）是层标识符的前缀。
这个值可能是[cudnn'，'singacpp'，'singacuda'，'singacl']之一，分别用cudnn库，Cpp，Cuda和OpenCL实现。 例如，CudnnConvolution层由'cudnn_convolution'标识; 'singacpp_convolution'用于卷积层; 有些层的实现只使用tensor函数，因此它们对底层设备是透明的。 对于这些层，它们将具有多个标识符，例如，singacpp_dropout，singacuda_dropout和singacl_dropout全都用于Dropout层。 此外，它还有一个额外的标识符'singa'，即'singa_dropout'也代表Dropout层。

引擎是大小写敏感的。每个python层将会用引擎属性创建正确的层。

---

### class singa.layer.Layer(name, conf=None, **kwargs)

基类：`object`

Python层的基类。
典型地，层实例的生命周期包括：

1. 构造层没有input_sample_shapes，转到2;用input_sample_shapes构建层，转到3
2. 调用setup来创建参数并设置其他元字段
3. 调用前向传播或访问层成员
4. 调用后向传播并获取参数完成更新

**参数：**
- **name (str)** – 层名

---

#### setup(in_shapes)

调用C++setup函数创建参数并设置元数据。

**参数：**
- **in_shapes** – 如果层接受单个输入tensor，则in_shapes是指定输入tensor形状的单个元组; 如果该层接受多个输入tensor（例如，concatenation层），则in_shapes是元组的元组，每个元组对于一个输入tensor

---

#### caffe_layer()

基于caffe层的配置创建一个SINGA层

---

#### get_output_sample_shape()

在setup以获得输出样本的形状后被调用

**返回值：** 单个输出tensor的元组；如果该层具有多个输出，则为元组列表

---

#### param_names()
**返回值：** 字符串列表，每个值代表一个参数tensor的名称

---

#### param_values()

返回参数值tensor。
参数tesnor不作为层成员存储。由于层设备的更改，cpp tensor可能会移动到diff设备上，这会导致不一致。

**返回值：** tensor列表，每个参数对应列表中的一个

---

#### forward(flag, x)

当前层的前向传播。

**参数：**
- **flag** – True (kTrain) for training (kEval); False for evaluating; other values for furture use.
- **x (Tensor or list<Tensor>)** – an input tensor if the layer is connected from a single layer; a list of tensors if the layer is connected from multiple layers.

**返回值：** 如果该层被连接在一个单独的层则返回tensor；如果被连接到多个层，则返回一个tensor列表

---

#### backward(flag, dy)

当前层的后向传播。

**参数：**
- **flag (int)** – 保留为以后使用
- **dy (Tensor or list<Tensor>)** – 与目标损失相对应的梯度tensor

**返回值：** <dx, <dp1, dp2..>>，dx是输入x的梯度，dpi是第i个参数的梯度

---

#### to_device(device)

将层状态tensor移至指定设备。

**参数：** 
- **device** – swig转换的设备，由singa.device创建

---

### class singa.layer.Dummy(name, input_sample_shape=None)

基类：`singa.layer.Layer`

一个虚拟层，仅用于向前/向后传递数据（输入/输出是单个tensor）。


#### forward(flag, x)

**返回值：** 输入x

#### backward(falg, dy)

**返回值：** dy，[]

---

### class singa.layer.Conv2D(name, nb_kernels, kernel=3, stride=1, border_mode='same', cudnn_prefer='fatest', data_format='NCHW', use_bias=True, W_specs=None, b_specs=None, pad=None, input_sample_shape=None)

基类：`singa.layer.Layer`

创建一个层做2D卷积。

**参数：**
- **nb_kernels (int)** – 输入tensor的通道（核）数
- **kernel** – 一个或一对整型数表示核的高和宽
- **stride** – 一个或一对整型数表示步长的高和宽
- **border_mode (string)** – 填充模式，不区分大小写，‘valid’ -> 在高和宽长度上补0 ‘same’ -> 填充核一半（下取整）数目的0，核必须是奇数
- **cudnn_prefer (string)** – 偏好的cudnn卷积算法，可以是‘fatest’, ‘autotune’, ‘limited_workspace’和‘no_workspace’
- **data_format (string)** – ‘NCHW’或‘NHWC’
- **use_bias (bool)** – True或False
- **pad** – 一个或一对整型数表示填充的高和宽
- **W_specs (dict)** – 用于指定权重矩阵的规格，字段包括代表参数名称的‘name’，代表学习速率乘数的'lr_mult，代表权重衰减乘数的''decay_mult'，代表初始化方法的'init'，其可以是'gaussian'，'uniform'，' xavier'，相应的初始化方法为'''std'，'mean'，'high'，'low'。TODO（wangwei）'clamp'为渐变约束，value为标量，'regularizer'为正规化，目前支持'l2'
- **b_specs (dict)** – 偏移向量的超参数，同W_specs类似
- **name (string)** – 层名
- **input_sample_shape** – 用于输入tensor形状的三元组，例如（通道，高度，宽度）或（高度，宽度，通道）

---

### class singa.layer.Conv1D(name, nb_kernels, kernel=3, stride=1, border_mode='same', cudnn_prefer='fatest', use_bias=True, W_specs={'init': 'Xavier'}, b_specs={'init': 'Constant', 'value': 0}, pad=None, input_sample_shape=None)

基类：`singa.layer.Conv2D`

构建1D卷积层。
大部分参数与Conv2D的参数相同，除了核，步长，填充值，这是一个标量而不是元组。 input_sample_shape是一个具有单个输入特征长度值的元组。

#### get_output_sample_shape()

---

### class singa.layer.Pooling2D(name, mode, kernel=3, stride=2, border_mode='same', pad=None, data_format='NCHW', input_sample_shape=None)

基类：`singa.layer.Layer`

2D池化层进行最大或平均池化。
所有的参数都与Conv2D相同，除了下面的参数。

**参数：**
- **mode** – 池化模式，model_pb2.PoolingConf.MAX或model_pb2.PoolingConf.AVE

---

### class singa.layer.MaxPooling2D(name, kernel=3, stride=2, border_mode='same', pad=None, data_format='NCHW', input_sample_shape=None)

基类: `singa.layer.Pooling2D`

---

### class singa.layer.AvgPooling2D(name, kernel=3, stride=2, border_mode='same', pad=None, data_format='NCHW', input_sample_shape=None)

基类: `singa.layer.Pooling2D`

---

### class singa.layer.MaxPooling1D(name, kernel=3, stride=2, border_mode='same', pad=None, data_format='NCHW', input_sample_shape=None)

基类: `singa.layer.MaxPooling2D`

get_output_sample_shape()

---

### class singa.layer.AvgPooling1D(name, kernel=3, stride=2, border_mode='same', pad=None, data_format='NCHW', input_sample_shape=None)

基类: `singa.layer.AvgPooling2D`

get_output_sample_shape()

---

### class singa.layer.BatchNormalization(name, momentum=0.9, beta_specs=None, gamma_specs=None, input_sample_shape=None)

基类：`singa.layer.Layer`

批量正则化。

**参数：**
- **momentum (float)** – 用于运行的均值和方差
- **beta_specs (dict)** – 字典，包括beta参数的字段：'name'参数名称'；lr_mult'学习速率乘数；'decay_mult'权重衰减乘数；'init'初始化方法；可以是'gaussian'，'uniform'和'xavier'，'std'，'mean'，'high'，'low'表示相应初始化方法；'clamp'表示梯度约束，值是标量；'regularizer'用于正则化，目前支持'l2'
- **gamma_specs (dict)** – 同beta_specs类似, 但用于gamma参数.
- **name (string)** – 层名
- **input_sample_shape (tuple)** – 整型数，至少一个

---

### class singa.layer.LRN(name, size=5, alpha=1, beta=0.75, mode='cross_channel', k=1, input_sample_shape=None)

基类：`singa.layer.Layer`

局部响应归一化。

**参数：**
- **size (int)** – 用于归一化的通道数.
- **mode (string)** – ‘cross_channel’
- **input_sample_shape (tuple)** – 3维元组，(channel, height, width)

---

### class singa.layer.Dense(name, num_output, use_bias=True, W_specs=None, b_specs=None, W_transpose=False, input_sample_shape=None)

基类：`singa.layer.Layer`

进行线性或放射变换，也被叫做内积或全连接层。

**参数：**
- **num_output (int)** – 输出特征长度
- **use_bias (bool)** – 转换后的特征向量是否加上偏移向量
- **W_specs (dict)** – 包含权值矩阵的字段：'name'参数名称'；lr_mult'学习速率乘数；'decay_mult'权重衰减乘数；'init'初始化方法；可以是'gaussian'，'uniform'和'xavier'，'std'，'mean'，'high'，'low'表示相应初始化方法；'clamp'表示梯度约束，值是标量；'regularizer'用于正则化，目前支持'l2'
- **b_specs (dict)** – 偏移向量的字段, 同W_specs类似
- **W_transpose (bool)** – 如果为真，输出为x*W.T+b
- **input_sample_shape (tuple)** – 输入特征长度

---

### class singa.layer.Dropout(name, p=0.5, input_sample_shape=None)

基类：`singa.layer.Layer`

Dropout层

**参数：**
- **p (float)** – 随机丢掉一个元素（即将其中设为0）的概率
- **name (string)** – 层名

---

### class singa.layer.Activation(name, mode='relu', input_sample_shape=None)

基类：`singa.layer.Layer`

激励层

**参数：**
- **name (string)** – 层名
- **mode (string)** – ‘relu’, ‘sigmoid’或 ‘tanh’
- **input_sample_shape (tuple)** – 单个样本的形状

---

### class singa.layer.Softmax(name, axis=1, input_sample_shape=None)

基类：`singa.layer.Layer`

采用SoftMax。

**参数：**
- **axis (int)** – 对[axis, -1)的数据逐个进行SoftMax
- **input_sample_shape (tuple)** – 单个样本的形状

---

### class singa.layer.Flatten(name, axis=1, input_sample_shape=None)

基类：`singa.layer.Layer`

将输入tensor重塑为一个矩阵。

**参数：**
- **axis (int)** – 根据指定维度将输入重塑为矩阵，[0,axis)作为行，[axis, -1)作为列
- **input_sample_shape (tuple)** – 单个样本的形状

---

### class singa.layer.Merge(name, input_sample_shape=None)

基类：`singa.layer.Layer`

对所有输入tensor求和。

**参数：**
- **input_sample_shape** – 输入样本的形状。所有样本的形状应该一致。

#### setup(in_shape)

#### get_output_sample_shape()

#### forward(flag, inputs)

通过求和合并所有输入tensor。
TODO(wangwei) 元素级别的合并操作。

**返回值：** 单个tensor，包含所有输入tensor的和

#### backward(flag, grad)

复制每个输入层的梯度tensor。

**参数：**
- **grad** - 梯度tensor

**返回值：** tensor列表，每个输入层对应其中一个

---

### class singa.layer.Split(name, num_output, input_sample_shape=None)

基类：`singa.layer.Layer`

生成输入tensor的多个副本。

**参数：**
- **num_output (int)** – 待生成的输出tensor数目
- **input_sample_shape()** – 包含一个整型数，代表输入样本特征大小

#### setup(in_shape)

#### get_output_sample_shape()

#### forward()

生成输入tensor的多个副本。

**参数：**
- **flag** – 没有用到
- **input** – 单个输入tensor

**返回值：** 输出tensor列表，每个对应输入的一个拷贝

#### backward()

对所有输入tensor求和得到单个输出tensor。

**参数：**
- **grad** - 梯度tensor

**返回值：** 一个tensor，代表所有输入梯度tensor的求和

---

### class singa.layer.Concat(name, axis, input_sample_shapes=None)

基类：`singa.layer.Layer`

将tensor竖直(axis=0)或水平(axis=1)拼接。目前仅支持2维tensor。

**参数：**
- **axis (int)** – 0表示拼接行; 1表示拼接列;
- **input_sample_shapes** – 样本形状的元组列表，每个对应一个输入样本的tensor

#### forward(flag, inputs)

拼接所有输入tensor。

**参数：**
- **flag** – 同Layer::forward()
- **input** – tensor列表

**返回值：** 一个拼接后的tensor


#### backward(flag, dy)

**参数：**
- **flag** – same as Layer::backward()
- **dy (Tensor)** – the gradient tensors of y w.r.t objective loss

**返回值：** 元组(dx, []), dx是tensor列表，对应输入的梯度；[]是空列表

---

### class singa.layer.Slice(name, axis, slice_point, input_sample_shape=None)

基类：`singa.layer.Layer`

将输入tensor沿竖直(axis=0)或水平(axis=1)分成多个子tensor。

**参数：**
- **axis (int)** – 0代表分割行; 1代表分割列;
- **slice_point (list)** – 沿着轴分割的位置；n-1个分割点对应n个子tensor；
- **input_sample_shape** – 输入样本tensor的形状

#### get_output_sample_shape()

#### forward(flag, x)

沿给定轴分割输入tensor。

**参数：**
- **flag** – 同Layer::forward()
- **x** – 单个输入tensor

**返回值：** 输出tensor列表

#### backward(flag, grads)

拼接所有梯度tensor以生成一个输出tensor。

**参数：**
- **flag** – 同Layer::backward()
- **grads** – tensor列表，每个对应一个分割的梯度tensor
**返回值：** 元组(dx, []), dx是一个tensor，对应原始输入的梯度；[]是空列表

---

### class singa.layer.RNN(name, hidden_size, rnn_mode='lstm', dropout=0.0, num_stacks=1, input_mode='linear', bidirectional=False, param_specs=None, input_sample_shape=None)

基类：`singa.layer.Layer`

递归层包含4个单元，即lstm, gru, tanh和relu。

**参数：**
- **hidden_size** – 隐含层特征大小，同所有层的堆栈。
- **rnn_mode** – 决定了RNN单元，可以是‘lstm’, ‘gru’, ‘tanh’和 ‘relu’。对于每种模式，可以参考cudnn手册。
- **num_stacks** – rnn层的堆栈数量。这不同于需要展开的序列长度。
- **input_mode** – 'linear'，通过线性变换将输入特征x转换成大小为hidden_size的特征向量；'skip'，仅要求输入特征大小等于hidden_size。
- **bidirection** – 对于双向RNN为真。
- **param_specs** – RNN参数的初始化配置。
- **input_sample_shape** – 包含一个整型数，代表输入样本的特征大小。

#### forward(flag, inputs)

**参数：**
- **flag** – True(kTrain) 代表训练；False(kEval)代表验证; 其他值用作以后使用。
- **<x1, x2,..xn, hx, cx>** – 其中，xi是输入(inputs,)第i个位置的tensor，它的形状是 (batch_size, input_feature_length); xi的batch_size必须大于xi + 1的大小; hx是初始隐藏状态，形状是（num_stacks * bidirection？2：1，batch_size，hidden_size）。 cx是与hy相同形状的初始细胞状态张量。 cx仅对lstm有效。 对于其他RNN，不存在cx。 hx和cx都可以是没有形状和数据的虚拟张量。
返回值：<y1，y2，... yn，hy，cy>，其中yi是第i个位置的输出张量，其形状是（batch_size，hidden_size *双向？2：1）。 hy是最终的隐藏状态张量。 cx是最终的细胞状态张量。 cx仅用于lstm。


#### backward(flag, grad)

**参数：**
- **flag** – 未来使用
- **<dy1, dy2,..dyn, dhy, dcy>** - 其中，dyi是(grad,) 第i个位置的梯度，它的形状是 (batch_size, hidden_size*bidirection?2 (i-th) – 1); dhy是最终隐藏状态的渐变，它的形状是（num_stacks *双向？2：1，batch_size，hidden_size）。dcy是最终单元状态的梯度。 cx仅对lstm有效，其他RNN不存在cx。 dhy和dcy都可以是没有形状和数据的虚拟tensor。

**返回值：** <dx1，dx2，... dxn，dhx，dcx>，其中dxi是第i个输入的梯度张量，它的形状是（batch_size，input_feature_length）。 dhx是初始隐藏状态的梯度。 dcx是初始单元状态的梯度，仅对lstm有效。

---

### class singa.layer.LSTM(name, hidden_size, dropout=0.0, num_stacks=1, input_mode='linear', bidirectional=False, param_specs=None, input_sample_shape=None)

基类：`singa.layer.RNN`

---

### class singa.layer.GRU(name, hidden_size, dropout=0.0, num_stacks=1, input_mode='linear', bidirectional=False, param_specs=None, input_sample_shape=None)

基类：`singa.layer.RNN`

---

#### singa.layer.get_layer_list()

返回包含所有支持图层的标识符（标签）的字符串列表

---
