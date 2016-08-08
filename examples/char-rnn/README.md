# Train Char-RNN using SINGA

Recurrent neural networks (RNN) are widely used for modelling sequential data,
e.g., natural language sentences. This example describe how to implement a RNN
application (or model) using SINGA's RNN layers.
We will use the [char-rnn](https://github.com/karpathy/char-rnn) modle as an
example, which trains over setences or
source code, with each character as an input unit. Particularly, we will train
a RNN using GRU over Linux kernel source code. After training, we expect to
generate meaningful code from the model.


## Instructions

* Compile and install SINGA. Currently the RNN implmentation depends on Cudnn V5.

* Prepare the dataset. Download the [kernel source code](http://cs.stanford.edu/people/karpathy/char-rnn/).
Other plain text files can also be used.

* Start the training,

    python train.py input_linux.txt

  Some hyper-parameters could be set through command line,

    python train.py -h


* Sample characters from the model by providing num of characters and the seed string.

    python sample.py 100 --seed '#include <std'
