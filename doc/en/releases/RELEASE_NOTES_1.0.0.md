#singa-incubating-1.0.0 Release Notes

---

SINGA is a general distributed deep learning platform for training big deep
learning models over large datasets. It is designed with an intuitive
programming model based on the layer abstraction. SINGA supports a wide variety
of popular deep learning models.

This release includes following features:

  * Core abstractions including Tensor and Device
      * [SINGA-207]  Update Tensor functions for matrices
      * [SINGA-205]  Enable slice and concatenate operations for Tensor objects
      * [SINGA-197]  Add CNMem as a submodule in lib/
      * [SINGA-196]  Rename class Blob to Block
      * [SINGA-194]  Add a Platform singleton
      * [SINGA-175]  Add memory management APIs and implement a subclass using CNMeM
      * [SINGA-173]  OpenCL Implementation
      * [SINGA-171]  Create CppDevice and CudaDevice
      * [SINGA-168]  Implement Cpp Math functions APIs
      * [SINGA-162]  Overview of features for V1.x
      * [SINGA-165]  Add cross-platform timer API to singa
      * [SINGA-167]  Add Tensor Math function APIs
      * [SINGA-166]  light built-in logging for making glog optional
      * [SINGA-164]  Add the base Tensor class


  * IO components for file read/write, network and data pre-processing
      * [SINGA-233]  New communication interface
      * [SINGA-215]  Implement Image Transformation for Image Pre-processing
      * [SINGA-214]  Add LMDBReader and LMDBWriter for LMDB
      * [SINGA-213]  Implement Encoder and Decoder for CSV
      * [SINGA-211]  Add TextFileReader and TextFileWriter for CSV files
      * [SINGA-210]  Enable checkpoint and resume for v1.0
      * [SINGA-208]  Add DataIter base class and a simple implementation
      * [SINGA-203]  Add OpenCV detection for cmake compilation
      * [SINGA-202]  Add reader and writer for binary file
      * [SINGA-200]  Implement Encoder and Decoder for data pre-processing



  * Module components including layer classes, training algorithms and Python binding
      * [SINGA-235]  Unify the engines for cudnn and singa layers
      * [SINGA-230]  OpenCL Convolution layer and Pooling layer
      * [SINGA-222]  Fixed bugs in IO
      * [SINGA-218]  Implementation for RNN CUDNN version
      * [SINGA-204]  Support the training of feed-forward neural nets
      * [SINGA-199]  Implement Python classes for SGD optimizers
      * [SINGA-198]  Change Layer::Setup API to include input Tensor shapes
      * [SINGA-193]  Add Python layers
      * [SINGA-192]  Implement optimization algorithms for Singa v1 (nesterove, adagrad, rmsprop)
      * [SINGA-191]  Add "autotune" for CudnnConvolution Layer
      * [SINGA-190]  Add prelu layer and flatten layer
      * [SINGA-189]  Generate python outputs of proto files
      * [SINGA-188]  Add Dense layer
      * [SINGA-187]  Add popular parameter initialization methods
      * [SINGA-186]  Create Python Tensor class
      * [SINGA-184]  Add Cross Entropy loss computation
      * [SINGA-183]  Add the base classes for optimizer, constraint and regularizer
      * [SINGA-180]  Add Activation layer and Softmax layer
      * [SINGA-178]  Add Convolution layer and Pooling layer
      * [SINGA-176]  Add loss and metric base classes
      * [SINGA-174]  Add Batch Normalization layer and Local Response Nomalization layer.
      * [SINGA-170]  Add Dropout layer and CudnnDropout layer.
      * [SINGA-169]  Add base Layer class for V1.0


  * Examples
      * [SINGA-232]  Alexnet on Imagenet
      * [SINGA-231]  Batchnormlized VGG model for cifar-10
      * [SINGA-228]  Add Cpp Version of Convolution and Pooling layer
      * [SINGA-227]  Add Split and Merge Layer and add ResNet Implementation

  * Documentation
      * [SINGA-239]  Transfer documentation files of v0.3.0 to github
      * [SINGA-238]  RBM on mnist
      * [SINGA-225]  Documentation for installation and Cifar10 example
      * [SINGA-223]  Use Sphinx to create the website

  * Tools for compilation and some utility code
      * [SINGA-229]  Complete install targets
      * [SINGA-221]  Support for Travis-CI
      * [SINGA-217]  build python package with setup.py
      * [SINGA-216]  add jenkins for CI support
      * [SINGA-212]  Disable the compilation of libcnmem if USE_CUDA is OFF
      * [SINGA-195]  Channel for sending training statistics
      * [SINGA-185]  Add CBLAS and GLOG detection for singav1
      * [SINGA-181]  Add NVCC supporting for .cu files
      * [SINGA-177]  Add fully cmake supporting for the compilation of singa_v1
      * [SINGA-172]  Add CMake supporting for Cuda and Cudnn libs
