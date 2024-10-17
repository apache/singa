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
from resizeimage import resizeimage

from singa import device
from singa import tensor
from singa import sonnx
import onnx
from utils import download_model, check_exist_or_download

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')


def preprocess(img):
    img = resizeimage.resize_cover(img, [224, 224], validate=False)
    img_ycbcr = img.convert('YCbCr')
    img_y_0, img_cb, img_cr = img_ycbcr.split()
    img_ndarray = np.asarray(img_y_0)
    img_4 = np.expand_dims(np.expand_dims(img_ndarray, axis=0), axis=0)
    img_5 = img_4.astype(np.float32) / 255.0
    return img_5, img_cb, img_cr


def get_image():
    # download image
    image_url = 'https://s3.amazonaws.com/model-server/inputs/kitten.jpg'
    img = Image.open(check_exist_or_download(image_url))
    return img


class MyModel(sonnx.SONNXModel):

    def __init__(self, onnx_model):
        super(MyModel, self).__init__(onnx_model)

    def forward(self, *x):
        y = super(MyModel, self).forward(*x)
        return y[0]

    def train_one_batch(self, x, y):
        pass


if __name__ == "__main__":

    url = 'https://github.com/onnx/models/raw/master/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.tar.gz'
    download_dir = '/tmp/'
    model_path = os.path.join(download_dir, 'super_resolution',
                              'super_resolution.onnx')

    logging.info("onnx load model...")
    download_model(url)
    onnx_model = onnx.load(model_path)

    # preprocess
    logging.info("preprocessing...")
    img = get_image()
    img_y, img_cb, img_cr = preprocess(img)
    # sg_ir = sonnx.prepare(onnx_model) # run without graph
    # y = sg_ir.run([img])

    logging.info("model compling...")
    dev = device.create_cuda_gpu()
    x = tensor.PlaceHolder(img_y.shape, device=dev)
    model = MyModel(onnx_model)
    model.compile([x], is_train=False, use_graph=True, sequential=True)

    # inference
    logging.info("model running...")
    x_batch = tensor.Tensor(device=dev, data=img_y)
    img_y = model.forward(x_batch)
    array_img_y = tensor.to_numpy(img_y)
    img_out_y = Image.fromarray(np.uint8((array_img_y[0] * 255.0).clip(0,
                                                                       255)[0]),
                                mode='L')

    # postprocess
    logging.info("postprocessing...")
    final_img = Image.merge("YCbCr", [
        img_out_y,
        img_cb.resize(img_out_y.size, Image.BICUBIC),
        img_cr.resize(img_out_y.size, Image.BICUBIC),
    ]).convert("RGB")
    final_img.show()
