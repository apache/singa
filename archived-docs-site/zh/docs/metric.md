# 度量(Metric)

该模块包含一组用于评估模型性能的度量类。 特定的度量类可以由C++的实现转换或直接使用Python实现。

示例用法：

```python
from singa import tensor
from singa import metric

x = tensor.Tensor((3, 5))
x.uniform(0, 1)  # randomly genearte the prediction activation
x = tensor.SoftMax(x)  # normalize the prediction into probabilities
y = tensor.from_numpy(np.array([0, 1, 3], dtype=np.int))  # set the truth

f = metric.Accuracy()
acc = f.evaluate(x, y)  # averaged accuracy over all 3 samples in x
```

---

### class singa.metric.Metric

基类：`object`

度量类的基类

封装C++度量类的子类可以使用继承的前向函数，并评估此基类的函数。 其他子类需要重写这些函数。 用户需要提供预测值和真实值来获取度量值。

---

#### forward(x, y)

为每个样本计算度量值

**参数：**
- **x (Tensor)** – 预测值，每行代表一个样本的预测值
- **y (Tensor)** – 真实值，每行代表一个样本的真实值

**返回值：** 浮点数组成的tensor，每个样本对应一个浮点数输出

---

#### evaluate(x, y)

计算样本的平均度量值

**参数：**
- **x (Tensor)** – 预测值，每列代表一个样本的预测值
- **y (Tensor)** – 真实值，每列代表一个样本的真实值

**返回值：** 浮点数组成的tensor，每个样本对应一个浮点数输出

---

### class singa.metric.Accuracy

基类：`singa.metric.Metric`

对于单标签预测任务，计算top-1精确度。它调用C++函数实现计算。

---
