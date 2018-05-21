# 优化器(Optimizer)

这个模块包含一系列用于模型参数更新的优化器。

示例用法：

```python
from singa import optimizer
from singa import tensor

sgd = optimizer.SGD(lr=0.01, momentum=0.9, weight_decay=1e-4)
p = tensor.Tensor((3,5))
p.uniform(-1, 1)
g = tensor.Tensor((3,5))
g.gaussian(0, 0.01)

sgd.apply(1, g, p, 'param')  # use the global lr=0.1 for epoch 1
sgd.apply_with_lr(2, 0.03, g, p, 'param')  # use lr=0.03 for epoch 2
```

-----

### class singa.optimizer.Optimizer(lr=None, momentum=None, weight_decay=None, regularizer=None, constraint=None)

基类：`object`

Python优化器类的基类。
典型地，优化器有如下作用：

1. 构建优化器
2. （可选）以参数名为注册每种参数
3. 使用优化器依照给定参数梯度及其他信息更新参数值

它的子类应该重写apply_with_lr函数已用于真实的参数更新。

**参数：**
- **lr (float)** – 学习速率
- **momentum (float)** – 动量
- **weight_decay (float)** – L2正则化系数，被排除于‘regularizer’之外
- **regularizer** –  Regularizer或RegularizerConf实例；如被设置，正则化将被用于apply_with_lr()。 用户也可以在优化器外部做正则化。
- **constraint** – Constraint或ConstraintConf实例；如被设置，正则化将被用于apply_with_lr()。 用户也可以在优化器外部做正则化。

-----

#### register(name, specs)

注册参数，包括为每个参数对象创建正则化和约束条件。 这里参数特定的正则化和约束比全局设置有更高优先级。 如果所有参数对于学习速率、正规化器和约束条件共享相同的设置，则不需要调用该函数。

**参数：**
- **name (str)** – 参数名
- **specs (ParamSpec)** – protobuf对象，包括正则化和约束条件、学习速率和权值衰减乘子。

-----

#### apply_regularizer_constraint(epoch, value, grad, name=None, step=-1)

如果可用，将采用正则化和约束条件。 如果全局正则化和参数特定的正则化都存在，会采用参数特定的正则化。

**参数：**
- **epoch (int)** – 训练的epoch ID （完整训练一遍训练数据称为一个epoch）
- **value (Tensor)** – 参数Tensor
- **grad (Tensor)** – 参数梯度Tensor
- **name (string)** – 参数名以索引到特定的规则（正则化或约束条件）
- **step (int)** – 每个epoch内的迭代ID

**返回值：** 更新后的梯度tensor

-----

#### apply_with_lr(epoch, lr, grad, value, name=None, step=-1)

如果grad非空，则根据给定学习速率更新参数。

子类优化器必须重写这个函数。如果grad为空，将不做任何操作。

**参数：**
- **epoch (int)** – 训练的epoch ID （完整训练一遍训练数据称为一个epoch）
- **lr (float)** – 学习速率
- **value (Tensor)** – 参数Tensor
- **grad (Tensor)** – 参数梯度Tensor
- **name (string)** – 参数名以索引到特定的规则（正则化或约束条件）
- **step (int)** – 每个epoch内的迭代ID
**返回值：** 更新后的参数值

-----

#### apply(epoch, grad, value, name=None, step=-1)

假设学习速率生成器配置完成，完成参数更新。 子类不需要重写这个函数。

**参数：**
- **epoch (int)** – 训练的epoch ID （完整训练一遍训练数据称为一个epoch）
- **value (Tensor)** – 参数Tensor
- **grad (Tensor)** – 参数梯度Tensor
- **name (string)** – 参数名以索引到特定的规则（正则化或约束条件）
- **step (int)** – 每个epoch内的迭代ID

**返回值：** 更新后的参数值

-----

### class singa.optimizer.SGD(lr=None, momentum=None, weight_decay=None, regularizer=None, constraint=None)

基类：`singa.optimizer.Optimizer`

原始的包含动量的随机梯度下降算法。 参数详见Optimizer基类。


#### apply_with_lr(epoch, lr, grad, value, name, step=-1)

-----

### class singa.optimizer.Nesterov(lr=None, momentum=0.9, weight_decay=None, regularizer=None, constraint=None)

基类：`singa.optimizer.Optimizer`

包含Nesterov动量的随机梯度下降算法。参数详见Optimizer基类。

#### apply_with_lr(epoch, lr, grad, value, name, step=-1)

-----

### class singa.optimizer.RMSProp(rho=0.9, epsilon=1e-08, lr=None, weight_decay=None, regularizer=None, constraint=None)

基类：`singa.optimizer.Optimizer`

RMSProp优化器。构造器参数请参考Optimizer基类。

**参数：**
- **rho (float)** – [0, 1]间的浮点数
- **epsilon (float)** – 很小的值，以避免数值误差

#### apply_with_lr(epoch, lr, grad, value, name, step=-1)

-----

### class singa.optimizer.AdaGrad(epsilon=1e-08, lr=None, weight_decay=None, lr_gen=None, regularizer=None, constraint=None)

基类：`singa.optimizer.Optimizer`

AdaGrad优化器。构造器参数请参考Optimizer基类。

**参数：** 
- **epsilon (float)** – 很小的值，以避免数值误差

#### apply_with_lr(epoch, lr, grad, value, name, step=-1)

-----

### class singa.optimizer.Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-08, lr=None, weight_decay=None, regularizer=None, constraint=None)

基类：`singa.optimizer.Optimizer`

Ada优化器。构造器参数请参考Optimizer基类。

**参数：**
- **beta_1 (float)** – 动量系数
- **beta_2 (float)** – 整合的梯度平方的系数
- **epsilon (float)** – 很小的值，以避免数值误差

#### apply_with_lr(epoch, lr, grad, value, name, step=-1)

更新一个参数对象

**参数：**
- **step (int)** – 累积训练迭代数，而不是当前迭代ID

-----

### class singa.optimizer.Regularizer

基类：`object`

Python参数梯度正则化的基类。

#### apply(epoch, value, grad, step=-1)

-----

### class singa.optimizer.CppRegularizer(conf)

基类：`singa.optimizer.Regularizer`

正则化的封装使用C++实现。

**参数：**
- **conf (RegularizerConf)** – protobuf配置信息

#### apply(epoch, value, grad, step=-1)

-----

### class singa.optimizer.L2Regularizer(coefficient)

基类：`singa.optimizer.Regularizer`

L2正则化。

**参数：**
- **coefficient (float)** – 正则化系数

#### apply(epoch, value, grad, step=-1)

-----

### class singa.optimizer.Constraint

基类：`object`

Python参数梯度约束的基类。

#### apply(epoch, value, grad, step=-1)

-----

### class singa.optimizer.CppConstraint(conf)

基类：`singa.optimizer.Constraint`

约束的封装使用C++实现。

**参数：**
- **conf (RegularizerConf)** – protobuf配置信息

#### apply(epoch, value, grad, step=-1)

-----

### class singa.optimizer.L2Constraint(threshold=None)

基类：`singa.optimizer.Constraint`

梯度缩放使得L2 norm小于给定阀值。

#### apply(epoch, value, grad, step=-1)

-----
