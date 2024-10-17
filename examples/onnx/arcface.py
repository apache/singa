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
# under the License.
#

import os
import numpy as np
from PIL import Image
from sklearn import preprocessing

from singa import device
from singa import tensor
from singa import sonnx
import onnx
from utils import download_model, check_exist_or_download

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


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


class MyModel(sonnx.SONNXModel):

    def __init__(self, onnx_model):
        super(MyModel, self).__init__(onnx_model)

    def forward(self, *x):
        y = super(MyModel, self).forward(*x)
        return y[0]

    def train_one_batch(self, x, y):
        pass


if __name__ == "__main__":

    download_dir = '/tmp'
    url = 'https://s3.amazonaws.com/onnx-model-zoo/arcface/resnet100/resnet100.tar.gz'
    model_path = os.path.join(download_dir, 'resnet100', 'resnet100.onnx')

    logging.info("onnx load model...")
    download_model(url)
    onnx_model = onnx.load(model_path)

    # inference demo
    logging.info("preprocessing...")
    img1, img2 = get_image()
    img1 = preprocess(img1)
    img2 = preprocess(img2)
    # sg_ir = sonnx.prepare(onnx_model) # run without graph
    # y = sg_ir.run([img1, img2])

    logging.info("model compling...")
    dev = device.create_cuda_gpu()
    x = tensor.Tensor(device=dev, data=np.concatenate((img1, img2), axis=0))
    model = MyModel(onnx_model)
    model.compile([x], is_train=False, use_graph=True, sequential=True)

    # verifty the test
    # from utils import load_dataset
    # inputs, ref_outputs = load_dataset(os.path.join('/tmp', 'resnet100', 'test_data_set_0'))
    # x_batch = tensor.Tensor(device=dev, data=inputs[0])
    # outputs = sg_ir.run([x_batch])
    # for ref_o, o in zip(ref_outputs, outputs):
    #     np.testing.assert_almost_equal(ref_o, tensor.to_numpy(o), 4)

    logging.info("model running...")
    y = model.forward(x)

    logging.info("postprocessing...")
    embedding = tensor.to_numpy(y)
    embedding = preprocessing.normalize(embedding)
    embedding1 = embedding[0]
    embedding2 = embedding[1]

    # Compute squared distance between embeddings
    dist = np.sum(np.square(embedding1 - embedding2))
    # Compute cosine similarity between embedddings
    sim = np.dot(embedding1, embedding2.T)
    # logging.info predictions
    logging.info('Distance = %f' % (dist))
    logging.info('Similarity = %f' % (sim))
