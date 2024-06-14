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

from singa import device
from singa import tensor
from singa import sonnx
from singa import autograd
import onnx

import sys

sys.path.append(os.path.dirname(__file__) + '/..')
from utils import download_model, check_exist_or_download

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')

from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')


def preprocess():
    text = "This film is so good"
    tokens = tokenizer.encode(text, add_special_tokens=True)
    tokens = np.array(tokens)
    return tokens.reshape([1, -1]).astype(np.float32)


class MyModel(sonnx.SONNXModel):

    def __init__(self, onnx_model):
        super(MyModel, self).__init__(onnx_model)

    def forward(self, *x):
        y = super(MyModel, self).forward(*x)
        return y[0]

    def train_one_batch(self, x, y):
        pass


if __name__ == "__main__":
    url = 'https://media.githubusercontent.com/media/onnx/models/master/text/machine_comprehension/roberta/model/roberta-sequence-classification-9.tar.gz'
    download_dir = '/tmp/'
    model_path = os.path.join(download_dir, 'roberta-sequence-classification-9',
                              'roberta-sequence-classification-9.onnx')

    logging.info("onnx load model...")
    download_model(url)
    onnx_model = onnx.load(model_path)

    # inference
    logging.info("preprocessing...")
    input_ids = preprocess()

    logging.info("model compling...")
    dev = device.get_default_device()
    x = tensor.Tensor(device=dev, data=input_ids)
    model = MyModel(onnx_model)

    # verifty the test
    # from utils import load_dataset
    # sg_ir = sonnx.prepare(onnx_model) # run without graph
    # inputs, ref_outputs = load_dataset(
    #     os.path.join('/tmp', 'roberta-sst-9', 'test_data_set_0'))
    # outputs = sg_ir.run(inputs)
    # for ref_o, o in zip(ref_outputs, outputs):
    #     np.testing.assert_almost_equal(ref_o, o, 4)

    logging.info("model running...")
    y = model.forward(x)
    y = autograd.reshape(y, y.shape[-2:])[-1, :]
    y = tensor.softmax(y)
    y = tensor.to_numpy(y)[0]
    y = np.argsort(y)[::-1]
    if (y[0] == 0):
        print("Prediction: negative")
    else:
        print("Prediction: positive")
