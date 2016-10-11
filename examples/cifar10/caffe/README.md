# Convert caffe model into singa model
This is a very basic implementation, you are welcomed to contribution to it. Now we support:
* Parse feed forward net
* Python language

# Implementation
1. Read proto file of caffe model using caffe.proto, and serialize it to string.
2. Parse the string use singa model.proto and get the layer config.
3. Setup each layer and add it to a feed forward net.

# Usage and example
1. Install `protobuf-compiler` by your favor package manager, e.g. `sudo apt-get install protobuf-compiler` for ubuntu
and `sudo yum install protobuf-compiler` for redhat/fedora.
2. Refer to [installation](http://singa.apache.org/en/docs/installation.html) to install pysinga.
2. Run `make` under this folder to compile `caffe.proto` into `caffe_pb2.py`.
3. Run `convert.py` to convert caffe model, e.g., if you want to convert alexnet for cifar10, you can run the following
script in your shell:
    `python convert.py cifar10_full_train_test.prototxt 3 32 32`
4. Usage of this script:
<pre><code>
    python convert.py -h         
    usage: convert.py [-h]
                      net_prototxt [input_sample_shape [input_sample_shape ...]]

    Caffe prototxt to SINGA model parameter converter. Note that only basic
    functions are implemented. You are welcomed to contribute to this file.

    positional arguments:
      net_prototxt        Path to the prototxt file for net in Caffe format
      input_sample_shape  The shape(in tuple) of input sample, example: 3 32 32

    optional arguments:
      -h, --help          show this help message and exit
</code></pre>
