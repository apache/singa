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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
from __future__ import print_function
from builtins import input
from builtins import range

import numpy as np
import os
import argparse
from PIL import Image

from singa import device
from singa import tensor
from singa import converter


# for debug: print norm of each layer
# net.verbose = True


def convert_model(prototxt, caffemodel):
    cvt = converter.CaffeConverter(net_proto=prototxt, param_path=caffemodel)
    model = cvt.create_net()
    cvt.convert_params(model)
    return model


def check_path(path):
    assert os.path.exists(path), 'File not found: ' + path


def read_image(img_path):
    # According to the VGG paper(Very Deep Convolutional Networks for
    # Large-Scale Image Recognition), the input images are zero-centered by
    # mean pixel(rather than mean image) substraction.
    mean_RGB = [123.68, 116.779, 103.939]

    img = Image.open(img_path)
    img = img.convert('RGB')
    resized = img.resize((224, 224))
    # order of axes: width,height,channel
    img_ary = np.asarray(resized, dtype=np.float32)
    img_ary -= mean_RGB
    # HWC -> CHW
    img_ary = np.swapaxes(img_ary, 0, 2)
    return np.asarray(img_ary)


def predict(net, dev, synset_list, topk=5):
    '''Predict the label of each image.

    Args:
        net, a pretrained neural net
        images, a batch of images [batch_size, 3, 32, 32], which have been
            pre-processed
        dev, the training device
        synset_list: the synset of labels
        topk, return the topk labels for each image.
    '''
    while True:
        img_path = eval(input("Enter input image path('quit' to exit): "))
        if img_path == 'quit':
            return
        if not os.path.exists(img_path):
            print('Path is invalid')
            continue
        img = read_image(img_path)
        x = tensor.from_numpy(img.astype(np.float32)[np.newaxis, :])
        x.to_device(dev)
        y = net.predict(x)
        y.to_host()
        prob = tensor.to_numpy(y)
        lbl = np.argsort(-prob[0])  # sort prob in descending order
        print([synset_list[lbl[i]] for i in range(topk)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert caffe vgg into singa. \
            This tool only supports caffe model from version as 29-Nov-2016. \
            You can use caffe tool to update previous model')
    parser.add_argument('model_txt', default='./vgg16.prototxt')
    parser.add_argument('model_bin', default='./vgg16.caffemodel')
    parser.add_argument('imgclass', default='./synset_words.txt')
    args = parser.parse_args()

    check_path(args.model_txt)
    check_path(args.model_bin)
    check_path(args.imgclass)

    model = convert_model(args.model_txt, args.model_bin)
    dev = device.get_default_device()
    model.to_device(dev)

    with open(args.imgclass, 'r') as fd:
        syn_li = [line.split(' ', 1)[1].strip('\n') for line in fd.readlines()]

    predict(model, dev, synset_list=syn_li, topk=5)
