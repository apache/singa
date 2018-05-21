# 软件架构

SINGA的软件架构包括三个主要部分，即核心（Core），输入输出（IO）和模型（Model）。 图1阐述了这些组件以及硬件。核心部分提供了内存管理和张量操作；输入输出包括从（向）磁盘和网络读取（写入）的类；模型部分为机器学习模型提供数据结构和算法支持，比如用于神经网络模型的层，用于通用机器学习模型的优化器/初始化/度量/损失函数等。


<img src="../_static/images/singav1-sw.png" align="center" width="500px"/>
<br/>
<span><strong>Figure 1 - SINGA V1 software stack.</strong></span>

## 核心

张量([Tensor](tensor.html))和设备([Device](device.html))是SINGA的两个核心抽象类。 Tensor代表了一个多维数组，存储了模型的变量并且为机器学习算法提供线性代数的操作，这些操作包括矩阵乘法和随机函数。每个Tensor实例被分配在一个设备实例上。 每个Device实例被创建在一台硬件设备上，如GPU或CPU核上。设备类用于管理张量所占用的内存以及在执行单元上执行张量操作，比如CPU线程或CUDA流。

依赖于硬件和编程语言，SINGA实现了以下特定的设备类：

* **CudaGPU** 代表一个Nvidia GPU。 执行单元是CUDA流。
* **CppCPU** 代表一个CPU。 执行单元是CPU线程。
* **OpenclGPU** 代表Nvidia和AMD的GPU。执行单元是CommandQueues。OpenCL和很多硬件设备兼容，比如FPGA和ARM，所以OpenclGPU可以扩展到其他设备上。

不同类型的设备使用不同编程语言书写用于张量操作的核函数，

* CppMath (tensor_math_cpp.h) 用Cpp实现了CppCPU的张量操作
* CudaMath (tensor_math_cuda.h) 用CUDA实现了CudaGPU的张量操作
* OpenclMath (tensor_math_opencl.h) 用OpenCL实现了OpenclGPU的张量操作

另外，不同类型的数据，比如float32和float16,可以通过加入相应的张量函数来支持。

典型地，用户将创建一个Device实例并把它传给多个Tensor实例。 当用户调用Tensor函数时，这些函数会自动唤起对应的实现(CppMath/CudaMath/OpenclMath)。 换句话说，Tensor操作的实现对用户是透明的。

大多数机器学习算法可以用（紧密的或稀疏的）Tensor表达。 所以，通过张量的抽象，SINGA可以运行很多模型，包括深度学习模型和其他传统机器学习模型。 

Tensor和Device的抽象化可以扩展通过不同编程语言以支持大量硬件设备。 一个新的硬件设备可以通过添加一个新的Device子类和实现相应的张量操作（xxxMath）加以支持。

基于速度和空间占用的优化可以被Device实现。 它管理着操作执行和内存的分配和释放。 更多的优化细节可以在[Device页面](device.html)看到。


## 模型

在Tensor和Device的抽象化之上，SINGA提供了更高级的类用于机器学习模型。

* [Layer](layer.html)和它的子类特别用于神经网络。 每个层为向前传递特征和向后传递梯度提供函数支持。 它们将复杂的操作封装起来使用户可以很容易创建神经网络连接一系列层。

* [Initializer](initializer.html)和它的子类为初始化模型参数（存储在Tesnor实例中）提供了可变的方法，包括Uniform,Gaussian等等。

* [Loss](loss.html)和它的子类定义了训练目标损失函数。与目标损失对应的计算损失值和计算梯度的函数都已被实现。 常见的损失函数包括平方差和交叉熵。

* [Metric](metric.html)和它的子类提供了评估模型性能的函数，比如精确度。

* [Optimizer](optimizer.html)和它的子类实现了如何利用参数梯度更新模型参数的方法，包括SGD, AdaGrad, RMSProp等等。


## 输入输出

输入输出包含数据加载，数据预处理和信息传递类。

* Reader和它的子类从磁盘文件加载字符串记录
* Writer和它的子类将字符串记录写到磁盘文件中
* Encoder和它的子类将Tensor实例编译成字符串记录
* Decoder和它的子类将字符串记录解码为Tensor实例
* Endpoint代表为消息传递提供函数的交互终端
* Message代表Endpoint实例间的交互消息。它会传递元数据和负载
