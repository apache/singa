## 下载 SINGA

* 要验证下载的tar.gz文件，请下载KEY和ASC文件，然后执行以下命令
 
        % gpg --import KEYS
        % gpg --verify downloaded_file.asc downloaded_file

  您还可以检查SHA512或MD5值以查看下载是否已完成

* v1.2.0 (2018年6月6日):
    * [Apache SINGA 1.2.0](http://www.apache.org/dyn/closer.cgi/incubator/singa/1.2.0/apache-singa-incubating-1.2.0.tar.gz)
      [\[SHA512\]](https://www.apache.org/dist/incubator/singa/1.2.0/apache-singa-incubating-1.2.0.tar.gz.sha512)
      [\[KEYS\]](https://www.apache.org/dist/incubator/singa/1.2.0/KEYS)
      [\[ASC\]](https://www.apache.org/dist/incubator/singa/1.2.0/apache-singa-incubating-1.2.0.tar.gz.asc)
    * [发行说明 1.2.0](releases/RELEASE_NOTES_1.2.0.html)
    * 新功能和主要更新,
        * 实现 autograd (目前支持MLP模型)
        * 升级 PySinga 以支持 Python 3
        * 为 Tensor 类添加 stride field
        * 讲 cuDNN 从 V5 升级到 V7
        * 为 ImageNet 分类任务添加 VGG, Inception V4, ResNet, 及 DenseNet 模型
        * 为 conda 包创建别名
        * 完善中文文档
        * 添加在 Windows 上运行 Singa 的说明
        * 更新编译, CI
        * 修复一些错误



* v1.1.0 (2017年2月12日):
     * [Apache SINGA 1.1.0](http://www.apache.org/dyn/closer.cgi/incubator/singa/1.1.0/apache-singa-incubating-1.1.0.tar.gz)
      [\[MD5\]](https://archive.apache.org/dist/incubator/singa/1.1.0/apache-singa-incubating-1.1.0.tar.gz.md5)
      [\[KEYS\]](https://archive.apache.org/dist/incubator/singa/1.1.0/KEYS)
      [\[ASC\]](https://archive.apache.org/dist/incubator/singa/1.1.0/apache-singa-incubating-1.1.0.tar.gz.asc)
    * [发行说明 1.1.0](releases/RELEASE_NOTES_1.1.0.html)
    * 新功能和主要更新,
        * 创建 Docker 镜像（CPU和GPU版本）
        * 为 SINGA 创建 Amazon AMI（CPU版本）
        * 与 Jenkins 集成以自动生成 Wheel 和 Debian 软件包（用于安装），并更新网站.
        * 增强 FeedFowardNet, 例如，用于调试的多输入和详细模式
        * 添加 Concat 和 Slice 层
        * 扩展 CrossEntropyLoss 以接受具有多个标签的实例
        * 添加 image_tool.py 与图像增强方法
        * 通过 Snapshot API 支持模型加载和保存
        * 在 Windows 上编译 SINGA 源代码
        * 用 SINGA 代码编译强制依赖库
        * 启用 S​​INGA 的 Java 绑定（基本）
        * 在检查点文件中添加版本 ID
        * 添加 Rafiki 工具包以提供 RESTFul API
        * 添加 Caffe 预训练的例子，包括 GoogleNet



* v1.0.0 (2016年9月8日):
    * [Apache SINGA 1.0.0](https://archive.apache.org/dist/incubator/singa/1.0.0/apache-singa-incubating-1.0.0.tar.gz)
      [\[MD5\]](https://archive.apache.org/dist/incubator/singa/1.0.0/apache-singa-incubating-1.0.0.tar.gz.md5)
      [\[KEYS\]](https://archive.apache.org/dist/incubator/singa//1.0.0/KEYS)
      [\[ASC\]](https://archive.apache.org/dist/incubator/singa/1.0.0/apache-singa-incubating-1.0.0.tar.gz.asc)
    * [发行说明 1.0.0](releases/RELEASE_NOTES_1.0.0.html)
    * 新功能和主要更新,
        * 用于支持更多机器学习模型的张量抽象.
        * 设备抽象运行在不同的硬件设备上，包括CPU，（Nvidia / AMD）GPU 和 FPGA（将在更高版本中测试）.
        * 用 cmake 替换 GNU autotool 进行编译.
        * 支持 Mac OS
        * 改进Python绑定，包括安装和编程
        * 更多深度学习模型, 包括 VGG 和 ResNet
        * 读取/写入文件和编码/解码数据的更多 IO 类
        * 直接基于 Socket 的新网络通信组件.
        * 包含 Dropout 和 RNN 层的 Cudnn V5.
        * 将网站制作工具从 maven 替换为 Sphinx
        * 整合 Travis-CI


* v0.3.0 (2016年4月20日):
    * [Apache SINGA 0.3.0](https://archive.apache.org/dist/incubator/singa/0.3.0/apache-singa-incubating-0.3.0.tar.gz)
      [\[MD5\]](https://archive.apache.org/dist/incubator/singa/0.3.0/apache-singa-incubating-0.3.0.tar.gz.md5)
      [\[KEYS\]](https://archive.apache.org/dist/incubator/singa/0.3.0/KEYS)
      [\[ASC\]](https://archive.apache.org/dist/incubator/singa/0.3.0/apache-singa-incubating-0.3.0.tar.gz.asc)
    * [发行说明 0.3.0](releases/RELEASE_NOTES_0.3.0.html)
    * 新功能和主要更新,
        * [在 GPU 集群上](v0.3.0/gpu.html) 可以在 GPU 集群上训练深度学习模型.
        * [Python 包装器的改进](v0.3.0/python.html) 使得配置工作变得很容易, 包括神经网络和 SGD 算法.
        * [新增 SGD 更新器](v0.3.0/updater.html), 包括 Adam, AdaDelta 和 AdaMax.
        * [安装](v0.3.0/installation.html) 具有较少的相关库， 对于单节点培训.
        * 在 CPU 和 GPU 上进行异构训练.
        * 支持 cuDNN V4.
        * 数据预读取.
        * 修复一些错误.



* v0.2.0 (2016年1月14日):
    * [Apache SINGA 0.2.0](https://archive.apache.org/dist/incubator/singa/0.2.0/apache-singa-incubating-0.2.0.tar.gz)
      [\[MD5\]](https://archive.apache.org/dist/incubator/singa/0.2.0/apache-singa-incubating-0.2.0.tar.gz.md5)
      [\[KEYS\]](https://archive.apache.org/dist/incubator/singa/0.2.0/KEYS)
      [\[ASC\]](https://archive.apache.org/dist/incubator/singa/0.2.0/apache-singa-incubating-0.2.0.tar.gz.asc)
    * [发行说明 0.2.0](releases/RELEASE_NOTES_0.2.0.html)
    * 新功能和主要更新,
        * [在 GPU 上进行训练](v0.2.0/gpu.html) 可以在具有多个 GPU 卡的单个节点上对复杂模型进行训练.
        * [混合神经网络划分](v0.2.0/hybrid.html) 同时支持数据和模型并行.
        * [Python 包装器](v0.2.0/python.html) 可以很容易地配置作业，包括神经网络和 SGD 算法.
        * [RNN 模型和 BPTT 算法](v0.2.0/general-rnn.html) 实现并支持基于 RNN 模型的应用，例如 GRU.
        * [云软件集成](v0.2.0/distributed-training.html) 包括 Mesos，Docker 和 HDFS.
        * 可视化神经网络结构和层信息，有助于调试.
        * 线性代数函数和针对 Blob 和原始数据指针的随机函数.
        * 新层，包括 Softmax 层，ArgSort 层，Dummy 层，RNN 层和 cuDNN 层
        * 更新 Layer 类以携带多个数据/梯度 Blob.
        * 通过加载预训练模型参数来提取新数据的特征和测试性能.
        * 为 IO 操作添加 Store 类.


* v0.1.0 (2015年10月8日):
    * [Apache SINGA 0.1.0](https://archive.apache.org/dist/incubator/singa/apache-singa-incubating-0.1.0.tar.gz)
      [\[MD5\]](https://archive.apache.org/dist/incubator/singa/apache-singa-incubating-0.1.0.tar.gz.md5)
      [\[KEYS\]](https://archive.apache.org/dist/incubator/singa/KEYS)
      [\[ASC\]](https://archive.apache.org/dist/incubator/singa/apache-singa-incubating-0.1.0.tar.gz.asc)
    * [Amazon EC2 image](https://console.aws.amazon.com/ec2/v2/home?region=ap-southeast-1#LaunchInstanceWizard:ami=ami-b41001e6)
    * [发行说明 0.1.0](releases/RELEASE_NOTES_0.1.0.html)
    * 主要功能包括,
        * 使用 GNU build 程序进行安装
        * 用 zookeeper 进行工作管理的脚本
        * 基于神经网络和层抽象的编程模型.
        * 基于 Worker，Server 和 Stub 的系统体系结构.
        * 训练三种不同类别的模型，即前馈模型，能量模型和 RNN 模型.
        * 使用 CPU 的同步和异步分布式训练框架
        * 检查点和恢复
        * 使用 gtest 进行单元测试

		
**免责声明（英文）**

Apache SINGA is an effort undergoing incubation at The Apache Software
Foundation (ASF), sponsored by the name of Apache Incubator PMC. Incubation is
required of all newly accepted projects until a further review indicates that
the infrastructure, communications, and decision making process have stabilized
in a manner consistent with other successful ASF projects. While incubation
status is not necessarily a reflection of the completeness or stability of the
code, it does indicate that the project has yet to be fully endorsed by the
ASF.
