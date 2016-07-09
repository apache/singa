# Training on GPU

---

Considering GPU is much faster than CPU for linear algebra operations,
it is essential to support the training of deep learning models (which involves
a lot of linear algebra operations) on GPU cards.
SINGA now supports training on a single node (i.e., process) with multiple GPU
cards. Training in a GPU cluster with multiple nodes is under development.

## Instructions

### Compilation
To enable the training on GPU, you need to compile SINGA with [CUDA](http://www.nvidia.com/object/cuda_home_new.html) from Nvidia,

    ./configure --enable-cuda --with-cuda=<path to cuda folder>

In addition, if you want to use the [CUDNN library](https://developer.nvidia.com/cudnn) for convolutional neural network
provided by Nvidia, you need to enable CUDNN,


    ./configure --enable-cuda --with-cuda=<path to cuda folder> --enable-cudnn --with-cudnn=<path to cudnn folder>

SINGA now supports CUDNN V3.0.


### Configuration

The job configuration for GPU training is similar to that for training on CPU.
There is one more field to configure, `gpu`, which indicate the device ID of
the GPU you want to use. The simplest configuration is


    # job.conf
    ...
    gpu: 0
    ...

This configuration will run the worker on GPU 0. If you want to launch multiple
workers, each on a separate GPU, you can configure it as

    # job.conf
    ...
    gpu: 0
    gpu: 2
    ...
    cluster {
      nworkers_per_group: 2
      nworkers_per_process: 2
    }

Using the above configuration, SINGA would partition each mini-batch evenly
onto two workers which run on GPU 0 and GPU 2 respectively. For more information
on running multiple workers in a single node, please refer to
[Training Framework](frameworks.html). Please be careful to configure the same number
of workers and number of `gpu`s. Otherwise some workers would run on GPU and the
rest would run on CPU. This kind of hybrid training is not well supported for now.


For some layers, their implementation is transparent to GPU/CPU, like the InnerProductLayer
GRULayer, ReLULayer, etc. Hence, you can use the same configuration for these layers to run
on GPU or CPU. For other layers, especially the layers involved in ConvNet, SINGA
uses different implementations for GPU and CPU. Particularly, the GPU version is
implemented using CUDNN library. To train a ConvNet on GPU, you configure the layers as

    layer {
      type: kCudnnConv
      ...
    }
    layer {
      type: kCudnnPool
      ...
    }

The [cifar10 example](cnn.html) and [Alexnet example](alexnet.html) have complete
configurations for ConvNet.

## Implementation details

SINGA implements the GPU training by assigning each worker a GPU device at the beginning
of training (by the Driver class). Then the work can call GPU functions and run them on the
assigned GPU. GPU is typically used for linear algebra computation in layer
functions, because GPU is good at such computation. There is a [Context]() singleton,
which stores the handles and random generators for each device. The layer code
should detect its running device and then call the CPU or GPU functions correspondingly.

To make the layer implementation easier
SINGA provides some linear algebra functions (in *math_blob.h*), which are transparent to the running
device for users. Internally, they query the Context singleton to get the device information
and call CPU or GPU to do the computation. Consequently, users can implement
layers without awareness of the underlying running device.

If the functionality cannot be implemented using SINGA provided functions in
*math_blob.h*, the layer code needs to handle the CPU and GPU devices explicitly
by querying the Context singleton.  For layers that cannot run on GPU, e.g.,
input/output layers and connection layers which have little computation but much
IO or network workload, there is no need to consider the GPU device.
When these layers are configured in a neural net, they will run on CPU (since
they don't call GPU functions).

