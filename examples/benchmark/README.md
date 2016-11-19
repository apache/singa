#Benchmark scripts

These scripts will test the efficiency of SINGA by training benchmark models pecified in
[convnet-benchmarks](https://github.com/soumith/convnet-benchmarks/tree/master/caffe/imagenet_winners)
over different devices (e.g., CPU and GPU).

To run them, create a python pip virtualenv or anaconda virtual environment as
guided by [this article](http://singa.apache.org/en/docs/installation.html#pip-and-anaconda-for-pysinga).
Then, execute the `run.py` as

    $ python run.py

Different models and devices could be tested, please refer to the command line help message,

    $ python run.py -h
