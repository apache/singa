---
name: DenseNet models on ImageNet
SINGA version: 1.1.1
SINGA commit:
license: https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py
---

# Image Classification using DenseNet


In this example, we convert DenseNet on [PyTorch](https://github.com/pytorch/vision/blob/master/torchvision/models/densenet.py)
to SINGA for image classification.

## Instructions

* Download one parameter checkpoint file (see below) and the synset word file of ImageNet into this folder, e.g.,

        $ wget https://s3-ap-southeast-1.amazonaws.com/dlfile/densenet/densenet-121.tar.gz
        $ wget https://s3-ap-southeast-1.amazonaws.com/dlfile/resnet/synset_words.txt
        $ tar xvf densenet-121.tar.gz

* Usage

        $ python serve.py -h

* Example

        # use cpu
        $ python serve.py --use_cpu --parameter_file densenet-121.pickle --depth 121 &
        # use gpu
        $ python serve.py --parameter_file densenet-121.pickle --depth 121 &

  The parameter files for the following model and depth configuration pairs are provided:
  [121](https://s3-ap-southeast-1.amazonaws.com/dlfile/densenet/densenet-121.tar.gz), [169](https://s3-ap-southeast-1.amazonaws.com/dlfile/densenet/densenet-169.tar.gz), [201](https://s3-ap-southeast-1.amazonaws.com/dlfile/densenet/densenet-201.tar.gz), [161](https://s3-ap-southeast-1.amazonaws.com/dlfile/densenet/densenet-161.tar.gz)

* Submit images for classification

        $ curl -i -F image=@image1.jpg http://localhost:9999/api
        $ curl -i -F image=@image2.jpg http://localhost:9999/api
        $ curl -i -F image=@image3.jpg http://localhost:9999/api

image1.jpg, image2.jpg and image3.jpg should be downloaded before executing the above commands.

## Details

The parameter files were converted from the pytorch via the convert.py program.

Usage:

    $ python convert.py -h
