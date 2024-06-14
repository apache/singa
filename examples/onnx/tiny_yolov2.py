#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under th

import os
import numpy as np
from PIL import Image, ImageDraw

from singa import device
from singa import tensor
from singa import sonnx
import onnx
from utils import download_model, check_exist_or_download

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')


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
                    color = clut[detectedClass]
                    x = x - w / 2
                    y = y - h / 2
                    draw.line((x, y, x + w, y), fill=color)
                    draw.line((x, y, x, y + h), fill=color)
                    draw.line((x + w, y, x + w, y + h), fill=color)
                    draw.line((x, y + h, x + w, y + h), fill=color)
                    draw.text((x, y), label[detectedClass], fill=color)
                    logging.info("bounding box: (%.2f, %.2f, %.2f, %.2f)" %
                                 (x, y, x + w, y + h))
                    logging.info('class=%s ; probability=%f' %
                                 (label[detectedClass],
                                  classes[detectedClass] * confidence))
    img.save("result.png")


class MyModel(sonnx.SONNXModel):

    def __init__(self, onnx_model):
        super(MyModel, self).__init__(onnx_model)

    def forward(self, *x):
        y = super(MyModel, self).forward(*x)
        return y[0]

    def train_one_batch(self, x, y):
        pass


if __name__ == "__main__":

    url = 'https://onnxzoo.blob.core.windows.net/models/opset_8/tiny_yolov2/tiny_yolov2.tar.gz'
    download_dir = '/tmp/'
    model_path = os.path.join(download_dir, 'tiny_yolov2', 'Model.onnx')

    logging.info("onnx load model...")
    download_model(url)
    onnx_model = onnx.load(model_path)

    # inference
    logging.info("preprocessing...")
    img = get_image()
    img = preprocess(img)
    # sg_ir = sonnx.prepare(onnx_model) # run without graph
    # y = sg_ir.run([img])

    logging.info("model compling...")
    dev = device.create_cuda_gpu()
    x = tensor.PlaceHolder(img.shape, device=dev)
    model = MyModel(onnx_model)
    model.compile([x], is_train=False, use_graph=True, sequential=True)

    # verifty the test
    # from utils import load_dataset
    # inputs, ref_outputs = load_dataset(os.path.join('/tmp', 'tiny_yolov2', 'test_data_set_0'))
    # x_batch = tensor.Tensor(device=dev, data=inputs[0])
    # outputs = sg_ir.run([x_batch])
    # for ref_o, o in zip(ref_outputs, outputs):
    #     np.testing.assert_almost_equal(ref_o, tensor.to_numpy(o), 4)

    logging.info("model running...")
    x = tensor.Tensor(device=dev, data=img)
    y = model.forward(x)

    logging.info("postprocessing...")
    out = tensor.to_numpy(y)[0]
    postprcess(out)
