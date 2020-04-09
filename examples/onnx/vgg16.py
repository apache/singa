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
logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')


def preprocess(img):
    img = img.resize((256, 256))
    img = img.crop((16, 16, 240, 240))
    img = np.array(img).astype(np.float32) / 255.
    img = np.rollaxis(img, 2, 0)
    for channel, mean, std in zip(range(3), [0.485, 0.456, 0.406],
                                  [0.229, 0.224, 0.225]):
        img[channel, :, :] -= mean
        img[channel, :, :] /= std
    img = np.expand_dims(img, axis=0)
    return img


def get_image_labe():
    # download label
    label_url = 'https://s3.amazonaws.com/onnx-model-zoo/synset.txt'
    with open(check_exist_or_download(label_url), 'r') as f:
        labels = [l.rstrip() for l in f]

    # download image
    image_url = 'https://s3.amazonaws.com/model-server/inputs/kitten.jpg'
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
    url = 'https://s3.amazonaws.com/onnx-model-zoo/vgg/vgg16/vgg16.tar.gz'
    download_dir = '/tmp/'
    model_path = os.path.join(download_dir, 'vgg16', 'vgg16.onnx')

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
    # inputs, ref_outputs = load_dataset(os.path.join('/tmp', 'vgg16', 'test_data_set_0'))
    # x_batch = tensor.Tensor(device=dev, data=inputs[0])
    # outputs = model.forward(x_batch)
    # for ref_o, o in zip(ref_outputs, outputs):
    #     np.testing.assert_almost_equal(ref_o, tensor.to_numpy(o), 4)

    # inference
    logging.info("preprocessing...")
    img, labels = get_image_labe()
    img = preprocess(img)

    logging.info("model running...")
    x_batch = tensor.Tensor(device=dev, data=img)
    y = model.forward(x_batch)

    logging.info("postprocessing...")
    y = tensor.softmax(y)
    scores = tensor.to_numpy(y)
    scores = np.squeeze(scores)
    a = np.argsort(scores)[::-1]
    for i in a[0:5]:
        logging.info('class=%s ; probability=%f' % (labels[i], scores[i]))
