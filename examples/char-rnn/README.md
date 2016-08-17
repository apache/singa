# Train Char-RNN over plain text

Recurrent neural networks (RNN) are widely used for modelling sequential data,
e.g., natural language sentences. This example describes how to implement a RNN
application (or model) using SINGA's RNN layers.
We will use the [char-rnn](https://github.com/karpathy/char-rnn) model as an
example, which trains over sentences or
source code, with each character as an input unit. Particularly, we will train
a RNN using GRU over Linux kernel source code. After training, we expect to
generate meaningful code from the model.


## Instructions

* Compile and install SINGA. Currently the RNN implementation depends on Cudnn with version >= 5.05.

* Prepare the dataset. Download the [kernel source code](http://cs.stanford.edu/people/karpathy/char-rnn/).
Other plain text files can also be used.

* Start the training,

        python train.py linux_input.txt

  Some hyper-parameters could be set through command line,

        python train.py -h

* Sample characters from the model by providing the number of characters to sample and the seed string.

        python sample.py 'model.bin' 100 --seed '#include <std'

  Please replace 'model.bin' with the path to one of the checkpoint paths.

