import os
import urllib.request
import gzip
import numpy as np
import codecs
import tarfile

from singa import device
from singa import tensor
from singa import opt
from singa import autograd
from singa import sonnx
import onnx
from onnx import version_converter, helper
import onnx.utils
from onnx.tools import update_model_dims


def load_model():
    model_url = 'https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.tar.gz'
    model_path = read_model_file(check_exist_or_download(model_url))
    return model_path


def read_model_file(path):
    download_dir = '/tmp/'
    model_name = 'mobilenetv2-1.0/./mobilenetv2-1.0.onnx'
    filename = os.path.join(download_dir, model_name)
    with tarfile.open(path, 'r') as t:
        t.extractall(path=download_dir)
    return filename


def check_exist_or_download(url):
    download_dir = '/tmp/'
    name = url.rsplit('/', 1)[-1]
    filename = os.path.join(download_dir, name)
    if not os.path.isfile(filename):
        print("Downloading %s" % url)
        urllib.request.urlretrieve(url, filename)
    return filename


def update_batch_size(onnx_model, batch_size):
    model_input = onnx_model.graph.input[0]
    model_input.type.tensor_type.shape.dim[0].dim_value = batch_size
    model_output = onnx_model.graph.output[0]
    model_output.type.tensor_type.shape.dim[0].dim_value = batch_size
    return onnx_model


if __name__ == "__main__":
    # create device
    dev = device.create_cuda_gpu()
    model_path = load_model()
    onnx_model = onnx.load(model_path)
    # set batch size
    onnx_model = update_batch_size(onnx_model, 64)
    sg_ir = sonnx.prepare(onnx_model, device=dev)
