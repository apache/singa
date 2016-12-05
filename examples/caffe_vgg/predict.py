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
import numpy as np
import os
import argparse
from PIL import Image

from singa import device
from singa import tensor
from singa import converter
from singa import layer
from singa import net

#for debug: print norm of each layer
#net.verbose = True


def convert_model(prototxt, caffemodel):
    cvt = converter.CaffeConverter(net_proto=prototxt, param_path=caffemodel)
    model = cvt.create_net()
    cvt.convert_params(model)
    return model


def check_path(path):
    assert os.path.exists(
        path), 'File not found: ' + path


def synset_list(sw_path):
    with open(sw_path, 'rb') as synsets:
        syn_li = []
        for line in synsets:
            syn_word = line.split(' ', 1)[1].strip('\n')
            syn_li.append(syn_word)
        return syn_li


def load_test_data(test_dir, mean):
    paths = os.listdir(test_dir)
    print paths
    test = []
    for path in paths:
        img = Image.open(os.path.join(test_dir, path))
        # BGR is the default model in caffe
        # convert RGB to BGR
        img = img.convert('RGB')
        r, g, b = img.split()
        img = Image.merge('RGB', (b, g, r))
        resized = img.resize((224, 224))
        # order of axes: width,height,channel
        img_ary = np.asarray(resized, dtype=np.float32)
        img_ary -= mean
        img_ary = np.swapaxes(img_ary, 0, 2)
        test.append(img_ary)
    return np.asarray(test)


def predict(net, images, dev, synset_list=None, topk=5):
    '''Predict the label of each image.

    Args:
        net, a pretrained neural net
        images, a batch of images [batch_size, 3, 32, 32], which have been
            pre-processed
        dev, the training device
        synset_list: the synset of labels
        topk, return the topk labels for each image.
    '''
    x = tensor.from_numpy(images.astype(np.float32))
    x.to_device(dev)
    y = net.predict(x)
    y.to_host()
    prob = tensor.to_numpy(y)
    labels = np.fliplr(np.argsort(prob))  # sort prob in descending order
    print labels[:, 0:topk]
    for i in range(labels.shape[0]):
        l = [synset_list[labels[i][j]] for j in range(topk)]
        print l


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert caffe vgg into singa. \
            This tool only supports caffe model in current version(29-Nov-2016). \
            You can use caffe tool to update previous model')
    parser.add_argument('model_txt', default='./vgg16.prototxt')
    parser.add_argument('model_bin', default='./vgg16.caffemodel')
    parser.add_argument('imgclass', default='./synset_words.txt')
    parser.add_argument('testdir', default='./test/')
    args = parser.parse_args()

    check_path(args.model_txt)
    check_path(args.model_bin)
    check_path(args.imgclass)
    check_path(args.testdir)

    model = convert_model(args.model_txt, args.model_bin)
    dev = device.get_default_device()
    model.to_device(dev)

    syn_li = synset_list(args.imgclass)

    # According to the VGG paper(Very Deep Convolutional Networks for
    # Large-Scale Image Recognition), the input images are zero-centered by
    # mean pixel(rather than mean image) substraction.
    mean_BGR =[103.939, 116.779, 123.68]
    test_images = load_test_data(args.testdir, mean_BGR)

    predict(model, test_images, dev, synset_list=syn_li, topk=5)
