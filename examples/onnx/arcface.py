import os
import urllib.request
import numpy as np
import tarfile
import glob
from PIL import Image
from sklearn import preprocessing

from singa import device
from singa import tensor
from singa import autograd
from singa import sonnx
import onnx
from onnx import version_converter, numpy_helper


def load_model():
    url = 'https://s3.amazonaws.com/onnx-model-zoo/arcface/resnet100/resnet100.tar.gz'
    download_dir = '/tmp/'
    filename = os.path.join(download_dir, 'resnet100', '.', 'resnet100.onnx')
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
    model_output = onnx_model.graph.output[0]
    model_output.type.tensor_type.shape.dim[0].dim_value = batch_size
    return onnx_model


def preprocess(img):
    w, h = img.size
    img = img.crop((0, (h - w) // 2, w, h - (h - w) // 2))
    img = img.resize((112, 112))
    img = np.array(img).astype(np.float32)
    img = np.rollaxis(img, 2, 0)
    img = np.expand_dims(img, axis=0)
    return img


def get_image():
    # download image
    img1 = Image.open(
        check_exist_or_download(
            'https://angus-doc.readthedocs.io/en/latest/_images/aurelien.jpg'))
    img2 = Image.open(
        check_exist_or_download(
            'https://angus-doc.readthedocs.io/en/latest/_images/gwenn.jpg'))
    return img1, img2


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

    # verifty the test dataset
    inputs, ref_outputs = load_dataset(os.path.join('/tmp', 'resnet100', 'test_data_set_0'))
    # x_batch = tensor.Tensor(device=dev, data=inputs[0])
    # outputs = model.forward(x_batch)
    # for ref_o, o in zip(ref_outputs, outputs):
    #     np.testing.assert_almost_equal(ref_o, tensor.to_numpy(o), 4)

    # inference
    img1, img2 = get_image()
    img1 = preprocess(img1)
    img2 = preprocess(img2)

    onnx_model = onnx.load(model_path)
    onnx_model = update_batch_size(onnx_model, 2)
    sg_ir = sonnx.prepare(onnx_model, device=dev)

    autograd.training = False
    model = Infer(sg_ir)

    x_batch = tensor.Tensor(device=dev,
                            data=np.concatenate((img1, img2), axis=0))
    y = model.forward(x_batch)
    embedding = tensor.to_numpy(y)
    embedding = preprocessing.normalize(embedding)
    embedding1 = embedding[0]
    embedding2 = embedding[1]

    # Compute squared distance between embeddings
    dist = np.sum(np.square(embedding1 - embedding2))
    # Compute cosine similarity between embedddings
    sim = np.dot(embedding1, embedding2.T)
    # Print predictions
    print('Distance = %f' % (dist))
    print('Similarity = %f' % (sim))
