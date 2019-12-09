# 设备(Device)

设备抽象代表了任何基于内存和计算单元的硬件设备。所有[Tensor操作](tensor.html)由寄宿的设备进行调度以执行。Tensor内存也由设备内存管理器进行管理。因此，内存优化和执行都由Device类进行实现。

## 特定设备

目前，SINGA有三种Device实现，

1. CudaGPU： 在Nvidia GPU卡上运行Cuda代码
2. CppCPU：在CPU上运行Cpp代码
3. OpenclGPU： GPU卡上运行OpenCL代码


## Python API

此脚本包括Device类和它的子类，用户可以调用singa::Device和它的方法。

---

#### singa.device.create_cuda_gpus(num)

创建一个列表的CudaGPU设备。

**参数：**
- **num(int)** - 创建的设备数目

**返回值：** 一个列表CudaGPU设备

---

#### singa.device.create_cuda_gpus_on(device_ids)

创建一个列表的CudaGPU设备。

**参数：**
- **device_ids(list)** - 一个列表的显卡ID

**返回值：** 一个列表CudaGPU设备

---

#### singa.device.get_default_device()

获取默认的CppCPU设备。

---

下面的代码展示了创建设备的例子：

``` python
from singa import device
cuda = device.create_cuda_gpu_on(0)  # use GPU card of ID 0
host = device.get_default_device()  # get the default host device (a CppCPU)
ary1 = device.create_cuda_gpus(2)  # create 2 devices, starting from ID 0
ary2 = device.create_cuda_gpus([0,2])  # create 2 devices on ID 0 and 2
```
