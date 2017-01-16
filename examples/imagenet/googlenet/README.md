---
name: GoogleNet on ImageNet
SINGA version: 1.0.1
SINGA commit: 8c990f7da2de220e8a012c6a8ecc897dc7532744
parameter_url: https://s3-ap-southeast-1.amazonaws.com/dlfile/bvlc_googlenet.tar.gz
parameter_sha1: 0a88e8948b1abca3badfd8d090d6be03f8d7655d
license: unrestricted https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet
---

# Image Classification using GoogleNet


In this example, we convert GoogleNet trained on Caffe to SINGA for image classification.

## Instructions

* Download the parameter checkpoint file into this folder

        $ wget https://s3-ap-southeast-1.amazonaws.com/dlfile/bvlc_googlenet.tar.gz
        $ tar xvf bvlc_googlenet.tar.gz

* Run the program

        # use cpu
        $ python serve.py -C &
        # use gpu
        $ python serve.py &

* Submit images for classification

        $ curl -i -F image=@image1.jpg http://localhost:9999/api
        $ curl -i -F image=@image2.jpg http://localhost:9999/api
        $ curl -i -F image=@image3.jpg http://localhost:9999/api

image1.jpg, image2.jpg and image3.jpg should be downloaded before executing the above commands.

## Details

We first extract the parameter values from [Caffe's checkpoint file](http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel) into a pickle version
After downloading the checkpoint file into `caffe_root/python` folder, run the following script

    # to be executed within caffe_root/python folder
    import caffe
    import numpy as np
    import cPickle as pickle

    model_def = '../models/bvlc_googlenet/deploy.prototxt'
    weight = 'bvlc_googlenet.caffemodel'  # must be downloaded at first
    net = caffe.Net(model_def, weight, caffe.TEST)

    params = {}
    for layer_name in net.params.keys():
        weights=np.copy(net.params[layer_name][0].data)
        bias=np.copy(net.params[layer_name][1].data)
        params[layer_name+'_weight']=weights
        params[layer_name+'_bias']=bias
        print layer_name, weights.shape, bias.shape

    with open('bvlc_googlenet.pickle', 'wb') as fd:
        pickle.dump(params, fd)

Then we construct the GoogleNet using SINGA's FeedForwardNet structure.
Note that we added a EndPadding layer to resolve the issue from discrepancy
of the rounding strategy of the pooling layer between Caffe (ceil) and cuDNN (floor).
Only the MaxPooling layers outside inception blocks have this problem.
Refer to [this](http://joelouismarino.github.io/blog_posts/blog_googlenet_keras.html) for more detials.
