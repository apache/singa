# 前馈网络

Neural net类用层来创建网络并提供可以获取网络信息（比如：参数）的函数。

示例用法：

```python
from singa import net as ffnet
from singa import metric
from singa import loss
from singa import layer
from singa import device

# create net and add layers
net = ffnet.FeedForwardNet(loss.SoftmaxCrossEntropy(), metric.Accuracy())
net.add(layer.Conv2D('conv1', 32, 5, 1, input_sample_shape=(3,32,32,)))
net.add(layer.Activation('relu1'))
net.add(layer.MaxPooling2D('pool1', 3, 2))
net.add(layer.Flatten('flat'))
net.add(layer.Dense('dense', 10))

# init parameters
for p in net.param_values():
    if len(p.shape) == 0:
        p.set_value(0)
    else:
        p.gaussian(0, 0.01)

# move net onto gpu
dev = device.create_cuda_gpu()
net.to_device(dev)

# training (skipped)

# do prediction after training
x = tensor.Tensor((2, 3, 32, 32), dev)
x.uniform(-1, 1)
y = net.predict(x)
print tensor.to_numpy(y)
```

---

### class singa.net.FeedForwardNet(loss=None, metric=None)

基类：`object`

#### to_device(dev)

将网络移至指定设备上，包括所有参数和中间数据。

---

#### add(lyr, src=None)

添加一个层到层列表中。

该功能将从src层获取样本形状以设置新添加的层。 对于第一层，它被设置在外部。 调用函数应确保层顺序的正确性。 如果src是None，最后一层是src层。 如果有多个src图层，则src是src层的列表。

**参数：**
- **lyr (Layer)** – 待添加的层
- **src (Layer)** – lyr层的父层

---

#### param_values()

返回所有参数的tensor列表。

---

#### param_specs()

返回所有参数的ParamSpec列表。

---

#### param_names()

返回所有参数名列表。。

---

#### train(x, y)

运行一次BP。
目前仅支持单输出层、单损失函数及度量方法的网络。 TODO(wangwei) 考虑多损失函数和多度量值。

**参数：**
- **x** – 输入数据，一个输入tensor或字典：层名->tensor
- **y** – 输入数据的标签，一个tensor

**返回值：** 参数梯度，损失函数和度量值

---

#### evaluate(x, y)

根据给定数据评估损失函数和度量值。目前仅支持单输出层、单损失函数及度量方法的网络。TODO(wangwei) 考虑多损失函数和多度量值。

**参数：**
- **x** – 输入数据，单个tensor或一个字典： 层名 -> tensor
- **y** – 输入数据的标签，单个tensor.

---

#### predict(x)

向前经每个层传递数据到输出层并获得输出值。
目前仅支持单输出层的网络。

**参数：**
- **x** - 输入数据，单个tesnor或一个字典： 层名 -> tensor

**返回值：** 单个输出tensor作为预测结果

---

#### topo_sort(layers, src_of_layer)

对所有层进行拓扑排序。
对于多输入层，将会保留输入层的顺序。

**参数：**
- **layers** – 层列表；同个层（如slice层）的多个输出层应该以正确的顺序加入，此功能将不会改变其顺序。
- **src_of_layer** – 字典: src层名 -> src层列表

**返回值：** 排序后层列表

---

#### forward(flag, x, output=[])

将输入经过每个层向前传递。
如果一个层具有来自其他层和来自x的输入，则来自x的数据在来自其他层的数据之前被排序，例如，如果层1->层2并且x [‘layer2’]具有数据，则输入层2展平，即[x ['layer2']，层1的输出]

**参数：**
- **flag** – True代表训练;False代表评估;也可以是model_pb2.kTrain或model_pb2.kEval或者其他未来可能使用的值。
- **x** – 一个tensor或一个字典：层名 -> tensor
- **output(list)** – 层名列表，将会和默认输出一起作为返回值

**返回值：** 如果只有一个输出层，返回输出tensor;否则返回字典：层名->输出tensor

---

#### backward()

运行向后传递

**返回值：** 所有参数的梯度tensor列表。

---

#### save(f, buffer_size=10, use_pickle=False)

用io/snapshot保存模型参数。

**参数：**
- **f** – 文件名
- **buffer_size** – 输入输出的大小(MB)，默认为10MB；请确保它比任何一个参数对象要大。 
- **use_pickle(boolean)** – 如果为真，将使用pickle保存；否则，将用protobuf做序列化，会占用较少空间。

---

#### load(f, buffer_size=10, use_pickle=False)

用io/snapshot加载模型参数。请参照save()的参数描述。

---
