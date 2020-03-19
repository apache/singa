import os
import urllib.request
import gzip
import numpy as np
import codecs
import tarfile
import warnings
import glob
from PIL import Image, ImageDraw

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


def preprocess(img):
    img = np.array(img).astype(np.float32)
    img = np.rollaxis(img, 2, 0)
    img = np.expand_dims(img, axis=0)
    return img


def get_image():
    image_url = 'https://raw.githubusercontent.com/simo23/tinyYOLOv2/master/person.jpg'
    img = Image.open(check_exist_or_download(image_url))
    img = img.resize((416, 416))
    return img


def postprcess(out):
    numClasses = 20
    anchors = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]

    def sigmoid(x, derivative=False):
        return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))

    def softmax(x):
        scoreMatExp = np.exp(np.asarray(x))
        return scoreMatExp / scoreMatExp.sum(0)

    clut = [(0, 0, 0), (255, 0, 0), (255, 0, 255), (0, 0, 255), (0, 255, 0),
            (0, 255, 128), (128, 255, 0), (128, 128, 0), (0, 128, 255),
            (128, 0, 128), (255, 0, 128), (128, 0, 255), (255, 128, 128),
            (128, 255, 128), (255, 255, 0), (255, 128, 128), (128, 128, 255),
            (255, 128, 128), (128, 255, 128)]
    label = [
        "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
        "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
        "pottedplant", "sheep", "sofa", "train", "tvmonitor"
    ]

    img = get_image()
    draw = ImageDraw.Draw(img)

    for cy in range(13):
        for cx in range(13):
            for b in range(5):
                channel = b * (numClasses + 5)
                tx = out[channel][cy][cx]
                ty = out[channel + 1][cy][cx]
                tw = out[channel + 2][cy][cx]
                th = out[channel + 3][cy][cx]
                tc = out[channel + 4][cy][cx]
                x = (float(cx) + sigmoid(tx)) * 32
                y = (float(cy) + sigmoid(ty)) * 32

                w = np.exp(tw) * 32 * anchors[2 * b]
                h = np.exp(th) * 32 * anchors[2 * b + 1]

                confidence = sigmoid(tc)

                classes = np.zeros(numClasses)
                for c in range(0, numClasses):
                    classes[c] = out[channel + 5 + c][cy][cx]

                classes = softmax(classes)
                detectedClass = classes.argmax()
                if 0.5 < classes[detectedClass] * confidence:
                    print(classes[detectedClass] * confidence)
                    print(label[detectedClass])
                    color = clut[detectedClass]
                    x = x - w / 2
                    y = y - h / 2
                    print(x, y, x + w, y)
                    draw.line((x, y, x + w, y), fill=color)
                    draw.line((x, y, x, y + h), fill=color)
                    draw.line((x + w, y, x + w, y + h), fill=color)
                    draw.line((x, y + h, x + w, y + h), fill=color)
                    draw.text((x, y),
                              label[detectedClass],
                              fill=color)
    img.save("result.png")


if __name__ == "__main__":
    # create device
    dev = device.create_cuda_gpu()
    model_path = load_model()
    onnx_model = onnx.load(model_path)

    # set batch size
    onnx_model = update_batch_size(onnx_model, 1)
    sg_ir = sonnx.prepare(onnx_model, device=dev)

    autograd.training = False
    model = Infer(sg_ir)

    # verifty the test dataset
    # inputs, ref_outputs = load_dataset(os.path.join('/tmp', 'tiny_yolov2', 'test_data_set_0'))
    # x_batch = tensor.Tensor(device=dev, data=inputs[0])
    # outputs = model.forward(x_batch)
    # for ref_o, o in zip(ref_outputs, outputs):
    #     np.testing.assert_almost_equal(ref_o, tensor.to_numpy(o), 4)

    # inference
    img = get_image()
    img = preprocess(img)
    x_batch = tensor.Tensor(device=dev, data=img)
    y = model.forward(x_batch)
    out = tensor.to_numpy(y)[0]
    postprcess(out)