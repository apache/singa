import os
import urllib.request
import gzip
import numpy as np
import codecs
import tarfile
import warnings
import glob

from singa import device
from singa import tensor
from singa import opt
from singa import autograd
from singa import sonnx
import onnx
from onnx import version_converter, helper, numpy_helper
import onnx.utils


def load_model():
    url = 'https://onnxzoo.blob.core.windows.net/models/opset_8/tiny_yolov2/tiny_yolov2.tar.gz'
    download_dir = '/tmp/'
    filename = os.path.join(download_dir, 'tiny_yolov2', '.', 'Model.onnx')
    with tarfile.open(check_exist_or_download(url), 'r') as t:
        t.extractall(path=download_dir)
    return filename


def load_dataset(test_data_dir):
    # Load inputs
    inputs = []
    inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
    for i in range(inputs_num):
        input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
        onnx_tensor = onnx.TensorProto()
        with open(input_file, 'rb') as f:
            onnx_tensor.ParseFromString(f.read())
        inputs.append(numpy_helper.to_array(onnx_tensor))

    # Load reference outputs
    ref_outputs = []
    ref_outputs_num = len(glob.glob(os.path.join(test_data_dir, 'output_*.pb')))
    for i in range(ref_outputs_num):
        output_file = os.path.join(test_data_dir, 'output_{}.pb'.format(i))
        onnx_tensor = onnx.TensorProto()
        with open(output_file, 'rb') as f:
            onnx_tensor.ParseFromString(f.read())
        ref_outputs.append(numpy_helper.to_array(onnx_tensor))
    return inputs, ref_outputs


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
    return onnx_model


class Infer:
    def __init__(self, sg_ir):
        self.sg_ir = sg_ir
        for idx, tens in sg_ir.tensor_map.items():
            # allow the tensors to be updated
            tens.requires_grad = True
            tens.stores_grad = True
            sg_ir.tensor_map[idx] = tens

    def forward(self, x):
        return sg_ir.run([x])[0]


if __name__ == "__main__":
    # create device
    dev = device.create_cuda_gpu()
    model_path = load_model()
    onnx_model = onnx.load(model_path)

    # set batch size
    onnx_model = update_batch_size(onnx_model, 1)
    sg_ir = sonnx.prepare(onnx_model, device=dev)

    # inference
    autograd.training = False
    print('The inference result is:')
    model = Infer(sg_ir)
    inputs, ref_outputs = load_dataset(os.path.join('/tmp', 'tiny_yolov2', 'test_data_set_0'))
    x_batch = tensor.Tensor(device=dev, data=inputs[0])
    outputs = model.forward(x_batch)

    # Compare the results with reference outputs.
    for ref_o, o in zip(ref_outputs, outputs):
        np.testing.assert_almost_equal(ref_o, tensor.to_numpy(o))
