# Train CNN over Cifar-10


Convolution neural network (CNN) is a type of feed-forward artificial neural
network widely used for image and video classification. In this example, we
will train three deep CNN models to do image classification for the CIFAR-10 dataset,

1. [AlexNet](https://code.google.com/p/cuda-convnet/source/browse/trunk/example-layers/layers-18pct.cfg)
the best validation accuracy (without data augmentation) we achieved was about 82%.

2. [VGGNet](http://torch.ch/blog/2015/07/30/cifar.html), the best validation accuracy (without data augmentation) we achieved was about 89%.
3. [ResNet](https://github.com/facebook/fb.resnet.torch), the best validation accuracy (without data augmentation) we achieved was about 83%.


## Instructions


### SINGA installation

Users can compile and install SINGA from source or install the Python version.
The code can ran on both CPU and GPU. For GPU training, CUDA and CUDNN (V4 or V5)
are required. Please refer to the installation page for detailed instructions.

### Data preparation

The binary Cifar-10 dataset could be downloaded by

    python download_data.py bin

The Python version could be downloaded by

    python download_data.py py

### Training

There are four training programs

1. train.py. The following command would train the VGG model using the python
version of the Cifar-10 dataset in 'cifar-10-batches-py' folder.

        python train.py vgg cifar-10-batches-py

    To train other models, please replace 'vgg' to 'alexnet' or 'resnet'. By default
    the training would run on a CudaGPU device, to run it on CppCPU, add an additional
    argument

        python train.py vgg cifar-10-batches-py  --use_cpu

2. alexnet.cc. It trains the AlexNet model using the CPP APIs on a CudaGPU,

        ./run.sh

3. alexnet-parallel.cc. It trains the AlexNet model using the CPP APIs on two CudaGPU devices.
The two devices run synchronously to compute the gradients of the mode parameters, which are
averaged on the host CPU device and then be applied to update the parameters.

        ./run-parallel.sh

4. vgg-parallel.cc. It train the VGG model using the CPP APIs on two CudaGPU devices similar to alexnet-parallel.cc.

### Prediction

predict.py includes the prediction function

        def predict(net, images, dev, topk=5)

The net is created by loading the previously trained model; Images consist of
a numpy array of images (one row per image); dev is the training device, e.g.,
a CudaGPU device or the host CppCPU device; topk labels of each image would be
returned.







