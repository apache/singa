# Train AlexNet over ImageNet

Convolution neural network (CNN) is a type of feed-forward neural
network widely used for image and video classification. In this example, we will
use a [deep CNN model](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
to do image classification against the ImageNet dataset.

## Instructions

### Compile SINGA

Please compile SINGA with CUDA, CUDNN and OpenCV. You can manually turn on the
options in CMakeLists.txt or run `ccmake ..` in build/ folder.

We have tested CUDNN V4 and V5 (V5 requires CUDA 7.5)

### Data download
* Please refer to step1-3 on [Instructions to create ImageNet 2012 data](https://github.com/amd/OpenCL-caffe/wiki/Instructions-to-create-ImageNet-2012-data)
  to download and decompress the data.
* You can download the training and validation list by
  [get_ilsvrc_aux.sh](https://github.com/BVLC/caffe/blob/master/data/ilsvrc12/get_ilsvrc_aux.sh)
  or from [Imagenet](http://www.image-net.org/download-images).

### Data preprocessing
* Assuming you have downloaded the data and the list.
  Now we should transform the data into binary files. You can run:

          sh create_data.sh

  The script will generate a test file(`test.bin`), a mean file(`mean.bin`) and
  several training files(`trainX.bin`) in the specified output folder.
* You can also change the parameters in `create_data.sh`.
  + `-trainlist <file>`: the file of training list;
  + `-trainfolder <folder>`: the folder of training images;
  + `-testlist <file>`: the file of test list;
  + `-testfolder <floder>`: the folder of test images;
  + `-outdata <folder>`: the folder to save output files, including mean, training and test files.
    The script will generate these files in the specified folder;
  + `-filesize <int>`: number of training images that stores in each binary file.

### Training
* After preparing data, you can run the following command to train the Alexnet model.

          sh run.sh

* You may change the parameters in `run.sh`.
  + `-epoch <int>`: number of epoch to be trained, default is 90;
  + `-lr <float>`: base learning rate, the learning rate will decrease each 20 epochs,
    more specifically, `lr = lr * exp(0.1 * (epoch / 20))`;
  + `-batchsize <int>`: batchsize, it should be changed regarding to your memory;
  + `-filesize <int>`: number of training images that stores in each binary file, it is the
    same as the `filesize` in data preprocessing;
  + `-ntrain <int>`: number of training images;
  + `-ntest <int>`: number of test images;
  + `-data <folder>`: the folder which stores the binary files, it is exactly the output
    folder in data preprocessing step;
  + `-pfreq <int>`: the frequency(in batch) of printing current model status(loss and accuracy);
  + `-nthreads <int>`: the number of threads to load data which feed to the model.
