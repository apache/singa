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
from PIL import Image

from singa import device
from singa import tensor
from singa import autograd
from singa import sonnx
import onnx
from utils import download_model, update_batch_size, check_exist_or_download

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def preprocess(img):
    input_shape = (1, 1, 64, 64)
    img = img.resize((64, 64), Image.ANTIALIAS)
    img_data = np.array(img).astype(np.float32)
    img_data = np.resize(img_data, input_shape)
    return img_data


def get_image_labe():
    labels = [
        'neutral', 'happiness', 'surprise', 'sadness', 'anger', 'disgust',
        'fear', 'contempt'
    ]
    # download image
    image_url = 'https://microsoft.github.io/onnxjs-demo/img/fear.8d1417fa.jpg'
    img = Image.open(check_exist_or_download(image_url))

    return img, labels


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

    url = 'https://onnxzoo.blob.core.windows.net/models/opset_8/emotion_ferplus/emotion_ferplus.tar.gz'
    download_dir = '/tmp/'
    model_path = os.path.join(download_dir, 'emotion_ferplus', 'model.onnx')

    logging.info("onnx load model...")
    download_model(url)
    onnx_model = onnx.load(model_path)

    # set batch size
    onnx_model = update_batch_size(onnx_model, 1)

    # prepare the model
    logging.info("prepare model...")
    dev = device.create_cuda_gpu()
    sg_ir = sonnx.prepare(onnx_model, device=dev)
    autograd.training = False
    model = Infer(sg_ir)

    # verifty the test
    # from utils import load_dataset
    # inputs, ref_outputs = load_dataset(os.path.join('/tmp', 'emotion_ferplus', 'test_data_set_0'))
    # x_batch = tensor.Tensor(device=dev, data=inputs[0])
    # outputs = model.forward(x_batch)
    # for ref_o, o in zip(ref_outputs, outputs):
    #     np.testing.assert_almost_equal(ref_o, tensor.to_numpy(o), 4)

    # inference
    logging.info("preprocessing...")
    img, labels = get_image_labe()
    img = preprocess(img)

    x_batch = tensor.Tensor(device=dev, data=img)

    logging.info("model running...")
    y = model.forward(x_batch)

    logging.info("postprocessing...")
    y = tensor.softmax(y)
    scores = tensor.to_numpy(y)
    scores = np.squeeze(scores)
    a = np.argsort(scores)[::-1]
    for i in a[0:5]:
        logging.info('class=%s ; probability=%f' % (labels[i], scores[i]))
