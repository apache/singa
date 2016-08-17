# Train a RBM model against MNIST dataset

This example is to train an RBM model using the
MNIST dataset. The RBM model and its hyper-parameters are set following
[Hinton's paper](http://www.cs.toronto.edu/~hinton/science.pdf)

## Running instructions

1. Download the pre-processed [MNIST dataset](https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz)

2. Start the training

        python train.py mnist.pkl.gz

By default the training code would run on CPU. To run it on a GPU card, please start
the program with an additional argument

        python train.py mnist.pkl.gz --use_gpu
