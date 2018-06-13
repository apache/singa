# 在MNIST数据集上训练RBM模型

这个例子使用MNIST数据集来训练一个RBM模型。RBM模型及其超参数参考[Hinton的论文](http://www.cs.toronto.edu/~hinton/science.pdf)中的设定。


## 操作说明

* 下载预处理的[MNIST数据集](https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz)。

* 开始训练，

        python train.py mnist.pkl.gz

  	默认情况下，训练代码将在CPU上运行。 要在GPU卡上运行它，请使用附加参数启动程序，

        python train.py mnist.pkl.gz --use_gpu


