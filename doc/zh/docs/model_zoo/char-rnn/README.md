# 在文本上训练Char-RNN

递归神经网络（RNN）被广泛用于建模顺序数据，例如自然语言句子。 本示例介绍如何使用SINGA的RNN层实现RNN应用程序（或模型）。 我们将使用[char-rnn](https://github.com/karpathy/char-rnn)模型作为示例，它将训练语句或源代码，并将每个字符作为输入单位。 特别是，我们将使用GRU在Linux内核源代码上训练一个RNN。 经过训练，我们希望从模型中生成有意义的代码。


## 操作说明

* 编译并安装SINGA。目前，RNN的实现是基于CuDNN（>=5.05）。.

* 准备数据集。下载[内核源代码](http://cs.stanford.edu/people/karpathy/char-rnn/)。其他文本数据也可被使用。 

* 开始训练，

        python train.py linux_input.txt

  一些超参数可以在命令行参数中设置，

        python train.py -h

* 通过提供要采样的字符数和种子字符串来从模型中采样字符。

        python sample.py 'model.bin' 100 --seed '#include <std'

  请用其中一个checkpoint路径替换'model.bin'的路径。

