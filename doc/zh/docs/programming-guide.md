# 编程指南

---

要提交一个训练作业，用户需要提供图1中的四个部分的配置：

  * [NeuralNet](neural-net.html) ：描述神经网络结构，包括每层的具体设置和层与层的连接关系；
  * [TrainOneBatch](train-one-batch.html) ：该算法需要根据不同的模型类别而定制;
  * [Updater](updater.html) ：定义服务器端更新参数的协议；
  * [Cluster Topology](distributed-training.html) ：指定服务器和工作者的分布式拓扑架构。

*初级用户指南* 将介绍如何利用内建层提交一个训练作业，而 *高级用户指南* 将详细介绍如何编写用户自己的主函数并注册自己实现的组件。此外，高级用户和初级用户对训练数据集的[处理](data.html)方式是相同的。

<img src="../_static/images/overview.png" align="center" width="400px"/>
<span><strong>图 1 - SINGA 概览</strong></span>



## 初级用户指南

用户可以使用SINGA提供的主函数提交训练作业。对于这种情况，用户必须在命令行中提供根据 [JobProto](../api/classsinga_1_1JobProto.html) 设置的作业配置文件，

    ./bin/singa-run.sh -conf <path to job conf> [-resume]

`-resume` 表示从上次的[检查点（checkpoint）](checkpoint.html)继续训练。
[MLP](mlp.html) 模型和 [CNN](cnn.html) 模型使用内建层提交训练作业。请阅读相关页面，查看它们的作业配置文件，这些页面会介绍每个组件配置的细节。

## 高级用户指南

如果用户的模型中包含一些自己定义的组件，比如[Updater](updater.html)，用户必须自己编写主函数注册这些组件，跟Hadoop的主函数类似。一般地，主函数应该

  * 初始化SINGA，如：设置日志；
  * 注册用户自定义组件；
  * 创建作业配置并传递给SINGA driver。

主函数示例

    #include "singa.h"
    #include "user.h"  // header for user code

    int main(int argc, char** argv) {
      singa::Driver driver;
      driver.Init(argc, argv);
      bool resume;
      // parse resume option from argv.

      // register user defined layers
      driver.RegisterLayer<FooLayer>(kFooLayer);
      // register user defined updater
      driver.RegisterUpdater<FooUpdater>(kFooUpdater);
      ...
      auto jobConf = driver.job_conf();
      //  update jobConf

      driver.Train(resume, jobConf);
      return 0;
    }

driver 类' `Init` 方法加载用户在命令行参数中 （`-conf <job conf>`）提供的作业配置文件（至少包含集群拓扑结构），并返回`jobConf`给用户，用户可更新和添加神经网络或者Updater的配置。如果定义了Layer、Updater、Worker或者Param的子类，用户需要通过driver为它们注册。最后，作业配置会被提交到driver，由driver启动训练。

将来我们会提供类似[keras](https://github.com/fchollet/keras) 的帮助工具，使作业配置更加简单。

用户需要使用SINGA库(*.libs/libsinga.so*)编译和链接自己的代码（如：layer的实现和主函数），得到可执行文件，如名为*mysinga* 的文件。执行以下命令启动该程序，用户需要将*mysinga* 和作业配置文件的路径传给 *./bin/singa-run.sh* 。

    ./bin/singa-run.sh -conf <path to job conf> -exec <path to mysinga> [other arguments]

[RNN application](rnn.html) 提供了一个完整的实现主函数训练特定RNN模型的例子。
