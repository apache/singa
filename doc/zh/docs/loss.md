# 损失(Loss)

损失模块包括一组训练损失实例。 有些是从C ++实现转换而来的，其余的都是直接使用python Tensor实现的。

示例用法：
```python
from singa import tensor
from singa import loss

x = tensor.Tensor((3, 5))
x.uniform(0, 1)  # randomly genearte the prediction activation
y = tensor.from_numpy(np.array([0, 1, 3], dtype=np.int))  # set the truth

f = loss.SoftmaxCrossEntropy()
l = f.forward(True, x, y)  # l is tensor with 3 loss values
g = f.backward()  # g is a tensor containing all gradients of x w.r.t l
```


---

### class singa.loss.Loss

基类：`object`

损失类的基类

封装C++损失类的子类可以使用此基类继承的forward，backward和evaluate函数。 其他子类需要重写这些函数

#### backward()

**返回值：** 与损失相对应的梯度

---

#### evaluate(flag, x, y)

**参数：**
- **flag (int)** – 必须是kEval
- **x (Tensor)** – 预测Tensor
- **y (Tensor)** – 真实Tensor

**返回值：** 所有样本的平均损失

---

#### forward(flag, x, y)

计算损失值

**参数：**
- **flag** – kTrain/kEval或布尔值。如果是kTrain/True，那么在下一次调用forward前会先调用backward计算梯度。
- **x (Tensor)** – 预测Tensor
- **y (Tensor)** – 真实Tensor, x.shape[0]必须和y.shape[0]相同

**返回值：** tensor，每个样本对应一个浮点型损失值

---

### class singa.loss.SoftmaxCrossEntropy

基类：`singa.loss.Loss`

此损失函数是SoftMax和交叉熵损失的结合。它通过SoftMax函数转换输入，然后根据真实值计算交叉熵损失。 对于每个样本，真实值可以是一个整数作为标签索引; 或二进制数组，指示标签分布。 因此，真实值可以是1维或2维tensor。 对于一批样品，数据/特征tensor可以是1维（对于单个样品）或2维（对于一组样本）。

---

### class singa.loss.SquaredError

基类：`singa.loss.Loss`

此损失用来衡量预测值和真实值之间的平方差。它通过Python Tensor操作实现。

---

#### backward()

计算与损失相对应变量的梯度。

**返回值：** x - y

---

#### evaluate(flag, x, y)

计算平均误差。

**返回值：** 浮点型数

---

#### forward(flag, x, y)

通过0.5 * ||x-y||^2计算损失。

**参数：**
- **flag (int)** – kTrain或kEval；如果是kTrain，那么在下一次调用forward前会先调用backward计算梯度。
- **x (Tensor)** – 预测Tensor
- **y (Tensor)** – 真实Tensor, 每个样本对应一个整型数, 取值为[0, x.shape[1])。

**返回值：** tensor，每个样本对应一个损失值

---
