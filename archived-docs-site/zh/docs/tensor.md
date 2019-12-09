# 张量(Tensor)

每个Tensor实例都是一个分配在特定Device实例上的多维数组。 Tensor实例存储了变量并提供了用户不可见的支持多种设备的代数操作。注意，用户需要确保除了拷贝之外的tensor操作都是在相同的设备上进行的。

Tensor的实现

SINGA有三种Tensor函数的实现，分别在不同设备上。

* `tensor_math_cpp.h` 用Cpp实现了CppCPU上的各种操作
* `tensor_math_cuda.h` 用Cuda (和cuBLAS)实现了CudaGPU上的各种操作
* `tensor_math_opencl.h` 用OpenCL实现了OpenclGPU上的各种操作


## PYTHON API

用法示例：

```python
import numpy as np
from singa import tensor
from singa import device

# create a tensor with shape (2,3), default CppCPU device and float32
x = tensor.Tensor((2, 3))
x.set_value(0.4)

# create a tensor from a numpy array
npy = np.zeros((3, 3), dtype=np.float32)
y = tensor.from_numpy(npy)

y.uniform(-1, 1)  # sample values from the uniform distribution

z = tensor.mult(x, y)  # gemm -> z of shape (2, 3)

x += z  # element-wise addition

dev = device.get_default_device()
x.to_device(dev)  # move the data to a gpu device

r = tensor.relu(x)

s = tensor.to_numpy(r)  # tensor -> numpy array
```

有两种类型的tensor函数:

**Tensor成员函数**

将会改变Tensor实例的状态

**Tensor模块化函数**

接受Tensor实例作为自变量以及返回Tensor实例

每个Tensor实例在读取数据前都必须做初始化

---

### class singa.tensor.Tensor(shape=None, device=None, dtype=0)

创建Py Tensor，封装了一个基于swig转换的CPP Tensor。
三个参数分别是Tensor的三个属性。

**参数：**
- **shape (list<int>)** – 一个列表的整形数据作为Tensor的形状。如果shape没有指定，将会创建一个伪Tensor。
- **device** – swig转化的使用设备模块化的Device实例。 如果为None，默认的CPU设备将会被使用。
- **dtype** – 数据类型。 目前，大多数操作仅支持kFloat32。

---

#### T()

浅拷贝。

**返回值：** 一个新Tensor，共享底层数据所占内存，但标记为该tensor的转置版本。

---

#### add_column(v)

对该Tensor每列加上一个tensor。

**参数：**
- **v (Tensor)** – 被作为一列加到原tensor的Tensor

---

#### add_row(v)

对该tensor每行加一个tensor。

**参数：**
- **v (Tensor)** – 被作为行加到原tensor的Tensor

---

#### bernoulli(p)

对每个元素，按照给定概率从0/1中取样。

**参数：**
- **p (float)** – 以概率p取样一个元素为1

---

#### clone()

**返回值：** 一个新Tensor，是待拷贝Tensor的深拷贝

---

#### copy()

调用singa::Tensor的拷贝构造器进行浅拷贝。

---

#### copy_data(t)

从另一个Tensor实例拷贝数据。

**参数：**
- **t (Tensor)** – 源Tensor

---

#### copy_from_numpy(np_array, offset=0)

从numpy数组中拷贝数据。

**参数：**
- **np_array** – 源numpy数组
- **offset (int)** – 目标偏移

---

#### deepcopy()

同clone()

**返回值：** 新Tensor

---

#### div_column(v)

将Tensor每列除以v。

**参数：**
- **v (Tensor)** – 1维tensor，和源tensor的列长相同

---

#### div_row(v)

将Tensor每行除以v。

**参数：**
- **v (Tensor)** – 1维tensor，和源tensor的行长相同

---

#### gaussian(mean, std)

按照高斯分布对每个元素采样。

**参数：**
- **mean (float)** – 分布的均值
- **std (float)** – 分布的标准差

---

#### is_empty()

**返回值：** 根据tensor的形状，如果是空的返回True

---

#### is_transpose()

**返回值：** 如果内部数据被转置则返回True，否则返回False

---

#### l1()

**返回值：** L1 norm

---

#### l2()

**返回值：** L2 norm

---

#### memsize()

- **返回值：** 被分配给该tensor的Bytes数

---

#### mult_column(v)

将tensor每列和v做元素级别乘法。

**参数：**
- **v (Tensor)** – 1维tensor，同源tensor列长等长

---

#### mult_row(v)

将tensor每行和v做元素级别乘法。

**参数：**
- **v (Tensor)** – 1维tensor，同源tensor行长等长

---

#### ndim()

**返回值：** tensor的维度

---

#### reset_like(t)

根据给定tensor重置源tensor形状，数据类型和设备。

**参数：**
- **t (Tensor)** – 需要重置的tensor

---

#### set_value(x)

设置所有元素值为给定值。

**参数：**
- **x(float)** - 待设定的值

---

#### size()

**返回值：** tensor中的元素个数

---

#### to_device(device)

将tensor中数据传到指定设备上。

**参数：**
- **device** - 从CudaGPU/CppCPU/OpenclGPU转换的swig设备

---

#### to_host()

将tensor数据传到默认的CppCPU设备上。

---

#### uniform(low, high)

从均匀分布中进行采样。

**参数：**
- **low (float)** – 下界
- **high (float)** – 上界

---

#### singa.tensor.abs(t)

**参数：**
- **t(Tensor)** - 输入tensor

**返回值：** 一个新tensor，其元素值为y=abs(x)，x是t中的元素

---

#### singa.tensor.add(lhs, rhs, ret=None)

元素级别加法。

**参数：**
- **lhs (Tensor)** – 左操作tensor
- **rhs (Tensor)** – 右操作tensor
- **ret (Tensor, optional)** – 如果不是空， 结果将被保存在其中；否则，一个新tensor会被创建以保存结果。

**返回值：** 新tensor

---

#### singa.tensor.add_column(alpha, v, beta, M)

将v加到M的每个列向量, 定义M一列为m，m=alpha * v + beta * m

**参数：**
- **alpha (float)** – v的系数
- **v (Tensor)** – 1维tensor
- **beta (float)** – M的系数
- **M (Tensor)** – 2维tensor

**返回值：** M

---

#### singa.tensor.add_row(alpha, v, beta, M)

将v加到M的每个行向量, 定义M一行为m，m=alpha * v + beta * m。

**参数：**
- **alpha (float)** – v的系数
- **v (Tensor)** – 1维tensor
- **beta (float)** – M的系数
- **M (Tensor)** – 2维tensor

**返回值：** M

---

#### singa.tensor.average(t, axis=None)

**参数：**
- **t (Tensor)** – 输入Tensor
- **axis (int, optional)** – 如果为空，取所有元素的平均值；否则，取给定维度的元素平均值。0表示列均值，1表示行均值。

**返回值：** 如果axis是空，返回一个float值；否则，返回一个新tensor

---

#### singa.tensor.axpy(alpha, x, y)

元素级别操作 y += alpha * x。

**参数：**
- **alpha (float)** – x的系数
- **x (Tensor)** – 被加的tensor
- **y (Tensor)** – 原tensor

**返回值：** y

---

#### singa.tensor.bernoulli(p, t)

对每个元素生成一个二进制位。

**参数：**
- **p (float)** – each element is 1 with probability p; and 0 with 1 - p
- **t (Tensor)** – the results are put into t

**返回值：** t

---

#### singa.tensor.copy_data_to_from(dst, src, size, dst_offset=0, src_offset=0)

将数据从一个tensor实例拷贝到另一个tensor实例。

**参数：**
- **dst (Tensor)** – 目标Tensor
- **src (Tensor)** – 源Tensor
- **size (int)** – 拷贝元素数目
- **dst_offset (int)** – 拷贝到dst元素在dst的起始偏移
- **src_offset (int)** – 待拷贝的元素在src中的起始偏移

---

#### singa.tensor.div(lhs, rhs, ret=None)

元素级别的除法。

**参数：**
- **lhs (Tensor)** – 左操作tensor
- **rhs (Tensor)** – 右操作tensor
- **ret (Tensor, optional)** – 如果非空，将把结果写入；否则，创建一个新tensor并将结果写入

**返回值：** 存有运算结果的tensor

---

#### singa.tensor.eltwise_mult(lhs, rhs, ret=None)

元素级别的乘法。

**参数：**
- **lhs (Tensor)** – 左操作tensor
- **rhs (Tensor)** – 右操作tensor
- **ret (Tensor, optional)** – 如果非空，将把结果写入；否则，创建一个新tensor并将结果写入

**返回值：** 保存运算结果的tensor

---

#### singa.tensor.exp(t)

**参数：**
- **t (Tensor)** – 输入tensor

**返回值：** 新tensor，其中元素为 y = exp(x)，x为t中元素

---

#### singa.tensor.from_numpy(np_array)

根据numpy数组的形状、数据类型和数值创建一个tensor。

**参数：**
- **np_array** – numpy数组

**返回值：** 分配在默认CppCPU设备上的tensor实例

---

#### singa.tensor.gaussian(mean, std, t)

按照给定高斯分布生成数值。

**参数：**
- **mean (float)** – 高斯分布的均值
- **std (float)** – 高斯分布的标准差
- **t (Tensor)** – 结果被存入t

**返回值：** t

---

#### singa.tensor.ge(t, x)

元素级别的比较，t >= x。

**参数：**
- **t (Tensor)** – 左操作数
- **x (Tensor or float)** – 右操作数

**返回值：** 0.0f 或 t[i] >= x[i] ? 1.0f:0.0f

**返回值类型：** tensor，每个元素为 t[i] >= x ? 1.0f

---

#### singa.tensor.gt(t, x)

元素级别的比较，t > x。

**参数：**
- **t (Tensor)** – 左操作tensor
- **x (Tensor or float)** – 右操作tensor或数

**返回值：** 0.0f 或 t[i] > x[i] ? 1.0f:0.0f

**返回值类型：** tensor，每个元素为 t[i] > x ? 1.0f

---

#### singa.tensor.le(t, x)

元素级别的比较，t <= x。

**参数：**
- **t (Tensor)** – 左操作tensor
- **x (Tensor or float)** – 右操作tensor或数

**返回值：** 0.0f 或 t[i] <= x[i] ? 1.0f:0.0f

**返回值类型：** tensor，每个元素为 t[i] <= x ? 1.0f

---

#### singa.tensor.lt(t, x)

元素级别的比较，t < x。

**参数：**
- **t (Tensor)** – 左操作tensor
- **x (Tensor or float)** – 右操作tensor或数

**返回值：** 0.0f 或 t[i] < x[i] ? 1.0f:0.0f

**返回值类型：** tensor，每个元素为 t[i] < x ? 1.0f

---

#### singa.tensor.log(t)

**参数：**
- **t (Tensor)** – 输入tensor

**返回值：** 一个新tensor，其元素值为y = log(x)，x是t中的元素

---

#### singa.tensor.mult(A, B, C=None, alpha=1.0, beta=0.0)

矩阵-矩阵或矩阵-向量乘法, 函数返回 C = alpha * A * B + beta * C。

**参数：**
- **A (Tensor)** – 2维Tensor
- **B (Tensor)** – 如果B是1维Tensor, 将调用GEMV做矩阵-向量乘法；否则将调用GEMM。
- **C (Tensor, optional)** – 存储结果；如果为空，将创建新tensor存储结果。
- **alpha (float)** – A * B 的系数
- **beta (float)** – C 的系数

**返回值：** 保存运算结果的tensor

---

#### singa.tensor.pow(t, x, out=None)

**参数：**
- **t (Tensor)** – 输入tensor
- **x (float or Tensor)** – 如果x是浮点数 y[i] = t[i]^x; 否则 y[i]= t[i]^x[i]
- **out (None or Tensor)** – 如果非空，将存入结果；否则，将创建一个新tensor保存结果。

**返回值：** 保存运算结果的tensor

---

#### singa.tensor.relu(t)

**参数：**
- **t (Tensor)** – 输入tensor

**返回值：** tensor，其中元素为 y = x 若x >0；否则y = 0，x为t中元素

---

#### singa.tensor.reshape(t, s)

改变tensor的形状。

**参数：**
- **t (Tensor)** – 待改变形状的tensor
- **s (list<int>)** – 新形状，体积和原tensor体积相同

**返回值：** 新tensor

---

#### singa.tensor.sigmoid(t)

**参数：**
- **t (Tensor)** – 输入tensor

**返回值：** tensor，其中元素为 y = sigmoid(x)，x为t中元素

---

#### singa.tensor.sign(t)

**参数：**
- **t (Tensor)** – 输入tensor

**返回值：** tensor，其中元素为 y = sign(x)，x为t中元素

---

#### singa.tensor.sizeof(dtype)

**返回值：** 依据core.proto中定义的SINGA数据类型，返回给定类型所占Byte数目

---

#### singa.tensor.softmax(t, out=None)

对tensor每行做SoftMax。

**参数：**
- **t (Tensor)** – 输入1维或2维tensor
- **out (Tensor, 可选)** – 如果非空，将存入结果

**返回值：** 保存操作结果的tensor

---

#### singa.tensor.sqrt(t)

**参数：**
- **t (Tensor)** – 输入tensor

**返回值：** tensor，其中元素为 y = sqrt(x)，x为t中元素

---

#### singa.tensor.square(t)

**参数：** 
- **t (Tensor)** – 输入tensor

**返回值：** tensor，其中元素为 y = x * x，x为t中元素

---

#### singa.tensor.sub(lhs, rhs, ret=None)

元素级别的减法。

**参数：**
- **lhs (Tensor)** – 左操作tensor
- **rhs (Tensor)** – 右操作tensor
- **ret (Tensor, 可选)** – 如果非空，将存入结果；否则，将创建一个新tensor保存

**返回值：** 存放结果的tensor

---

#### singa.tensor.sum(t, axis=None)

在给定的维度上求和。

**参数：**
- **t (Tensor)** – 输入Tensor
- **axis (int, 可选)** – 如果为空，将对所有元素求和；如果给定数值，将沿给定维度求和，比如：0 - 按列求和；1 - 按行求和。

**返回值：** 如果是对整体求和，返回一个浮点数；否则返回tensor

---

#### singa.tensor.sum_columns(M)

按列求和。

**参数：**
- **M (Tensor)** – 输入的2维tensor

**返回值：** 产生求和结果的tensor

---

#### singa.tensor.sum_rows(M)

按行求和。

**参数：**
- **M (Tensor)** – 输入的2维tensor

**返回值：** 产生求和结果的tensor

---

#### singa.tensor.tanh(t)

**参数：**
- **t (Tensor)** – 输入tensor

**返回值：**tensor，其中元素为 y = tanh(x)，x为t中元素

---

#### singa.tensor.to_host(t)

将数据拷贝到host设备上。

---

#### singa.tensor.to_numpy(t)

拷贝tensor数据到numpy数组。

**参数：**
- **t (Tensor)** – 输入tensor

**返回值：** numpy数组

---

#### singa.tensor.uniform(low, high, t)

按照均匀分布生成数值。

**参数：**
- **low (float)** – 下界
- **hight (float)** – 上届
- **t (Tensor)** – 结果存入t

**返回值：** t

---
