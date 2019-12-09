<!--
    Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.
-->
# Quickstart - Cifar10 example
Convolution neural network (CNN) is a type of feed-forward artificial neural network widely used for image classification. In this example, we will use a deep CNN model to do image classification for the [CIFAR10 dataset](http://www.cs.toronto.edu/~kriz/cifar.html).

## Running instructions for CPP version
Please refer to [Installation](installation.html) page for how to install SINGA. Currently, we CNN requires CUDNN, hence both CUDA and CUDNN should be installed and SINGA should be compiled with CUDA and CUDNN.

The Cifar10 dataset could be downloaded by running

    # switch to cifar10 directory
    $ cd ../examples/cifar10
    # download data for CPP version
    $ python download_data.py bin

'bin' is for downloading binary version of Cifar10 data.

During downloading, you should see the detailed output like

     Downloading CIFAR10 from http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
     The tar file does exist. Extracting it now..
     Finished!

Now you have prepared the data for this Cifar10 example, the final step is to execute the `run.sh` script,

    # in SINGA_ROOT/examples/cifar10/
    $ ./run.sh

You should see the detailed output as follows: first read the data files in order, show the statistics of training and testing data, then show the details of neural net structure with some parameter information, finally illustrate the performance details during training and validation process. The number of epochs can be specified in `run.sh` file.

    Start training
    Reading file cifar-10-batches-bin/data_batch_1.bin
    Reading file cifar-10-batches-bin/data_batch_2.bin
    Reading file cifar-10-batches-bin/data_batch_3.bin
    Reading file cifar-10-batches-bin/data_batch_4.bin
    Reading file cifar-10-batches-bin/data_batch_5.bin
    Reading file cifar-10-batches-bin/test_batch.bin
    Training samples = 50000, Test samples = 10000
    conv1(32, 32, 32, )
    pool1(32, 16, 16, )
    relu1(32, 16, 16, )
    lrn1(32, 16, 16, )
    conv2(32, 16, 16, )
    relu2(32, 16, 16, )
    pool2(32, 8, 8, )
    lrn2(32, 8, 8, )
    conv3(64, 8, 8, )
    relu3(64, 8, 8, )
    pool3(64, 4, 4, )
    flat(1024, )
    ip(10, )
    conv1_weight : 8.09309e-05
    conv1_bias : 0
    conv2_weight : 0.00797731
    conv2_bias : 0
    conv3_weight : 0.00795888
    conv3_bias : 0
    ip_weight : 0.00798683
    ip_bias : 0
    Messages will be appended to an existed file: train_perf
    Messages will be appended to an existed file: val_perf
    Epoch 0, training loss = 1.828369, accuracy = 0.329420, lr = 0.001000
    Epoch 0, val loss = 1.561823, metric = 0.420600
    Epoch 1, training loss = 1.465898, accuracy = 0.469940, lr = 0.001000
    Epoch 1, val loss = 1.361778, metric = 0.513300
    Epoch 2, training loss = 1.320708, accuracy = 0.529000, lr = 0.001000
    Epoch 2, val loss = 1.242080, metric = 0.549100
    Epoch 3, training loss = 1.213776, accuracy = 0.571620, lr = 0.001000
    Epoch 3, val loss = 1.175346, metric = 0.582000

The training details are stored in `train_perf` file in the same directory and the validation details in `val_perf` file.


## Running instructions for Python version
To run CNN example in Python version, we need to compile SINGA with Python binding,

    $ mkdir build && cd build
    $ cmake -DUSE_PYTHON=ON ..
    $ make

Now download the Cifar10 dataset,

    # switch to cifar10 directory
    $ cd ../examples/cifar10
    # download data for Python version
    $ python download_data.py py

During downloading, you should see the detailed output like

     Downloading CIFAR10 from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
     The tar file does exist. Extracting it now..
     Finished!

Then execute the `train.py` script to build the model

    $ python train.py

You should see the output as follows including the details of neural net structure with some parameter information, reading data files, and the performance details during training and testing process.

    (32L, 32L, 32L)
    (32L, 16L, 16L)
    (32L, 16L, 16L)
    (32L, 16L, 16L)
    (32L, 16L, 16L)
    (32L, 16L, 16L)
    (32L, 8L, 8L)
    (32L, 8L, 8L)
    (64L, 8L, 8L)
    (64L, 8L, 8L)
    (64L, 4L, 4L)
    (1024L,)
    Start intialization............
    conv1_weight gaussian 7.938460476e-05
    conv1_bias constant 0.0
    conv2_weight gaussian 0.00793507322669
    conv2_bias constant 0.0
    conv3_weight gaussian 0.00799657031894
    conv3_bias constant 0.0
    dense_weight gaussian 0.00804364029318
    dense_bias constant 0.0
    Loading data ..................
    Loading data file cifar-10-batches-py/data_batch_1
    Loading data file cifar-10-batches-py/data_batch_2
    Loading data file cifar-10-batches-py/data_batch_3
    Loading data file cifar-10-batches-py/data_batch_4
    Loading data file cifar-10-batches-py/data_batch_5
    Loading data file cifar-10-batches-py/test_batch
    Epoch 0
    training loss = 1.881866, training accuracy = 0.306360 accuracy = 0.420000
    test loss = 1.602577, test accuracy = 0.412200
    Epoch 1
    training loss = 1.536011, training accuracy = 0.441940 accuracy = 0.500000
    test loss = 1.378170, test accuracy = 0.507600
    Epoch 2
    training loss = 1.333137, training accuracy = 0.519960 accuracy = 0.520000
    test loss = 1.272205, test accuracy = 0.540600
    Epoch 3
    training loss = 1.185212, training accuracy = 0.574120 accuracy = 0.540000
    test loss = 1.211573, test accuracy = 0.567600

This script will call `alexnet.py` file to build the alexnet model. After the training is finished, SINGA will save the model parameters into a checkpoint file `model.bin` in the same directory. Then we can use this `model.bin` file for prediction.

    $ python predict.py
