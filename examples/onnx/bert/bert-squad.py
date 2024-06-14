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
import zipfile
import numpy as np
import json

from singa import device
from singa import tensor
from singa import sonnx
import onnx
import tokenization
from run_onnx_squad import read_squad_examples, convert_examples_to_features, RawResult, write_predictions

import sys

sys.path.append(os.path.dirname(__file__) + '/..')
from utils import download_model, check_exist_or_download

import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')

max_answer_length = 30
max_seq_length = 256
doc_stride = 128
max_query_length = 64
n_best_size = 20
batch_size = 1


def load_vocab():
    url = 'https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip'
    download_dir = '/tmp/'
    filename = os.path.join(download_dir, 'uncased_L-12_H-768_A-12', '.',
                            'vocab.txt')
    with zipfile.ZipFile(check_exist_or_download(url), 'r') as z:
        z.extractall(path=download_dir)
    return filename


def preprocess():
    vocab_file = load_vocab()
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file,
                                           do_lower_case=True)
    predict_file = os.path.join(os.path.dirname(__file__), 'inputs.json')
    # print content
    with open(predict_file) as json_file:
        test_data = json.load(json_file)
        print("The input is:", json.dumps(test_data, indent=2))

    eval_examples = read_squad_examples(input_file=predict_file)

    # Use convert_examples_to_features method from run_onnx_squad to get parameters from the input
    input_ids, input_mask, segment_ids, extra_data = convert_examples_to_features(
        eval_examples, tokenizer, max_seq_length, doc_stride, max_query_length)
    return input_ids, input_mask, segment_ids, extra_data, eval_examples


def postprocess(eval_examples, extra_data, all_results):
    output_dir = 'predictions'
    os.makedirs(output_dir, exist_ok=True)
    output_prediction_file = os.path.join(output_dir, "predictions.json")
    output_nbest_file = os.path.join(output_dir, "nbest_predictions.json")
    write_predictions(eval_examples, extra_data, all_results, n_best_size,
                      max_answer_length, True, output_prediction_file,
                      output_nbest_file)

    # print results
    with open(output_prediction_file) as json_file:
        test_data = json.load(json_file)
        print("The result is:", json.dumps(test_data, indent=2))


class MyModel(sonnx.SONNXModel):

    def __init__(self, onnx_model):
        super(MyModel, self).__init__(onnx_model)

    def forward(self, *x):
        y = super(MyModel, self).forward(*x)
        return y

    def train_one_batch(self, x, y):
        pass


if __name__ == "__main__":

    url = 'https://media.githubusercontent.com/media/onnx/models/master/text/machine_comprehension/bert-squad/model/bertsquad-10.tar.gz'
    download_dir = '/tmp/'
    model_path = os.path.join(download_dir, 'download_sample_10',
                              'bertsquad10.onnx')

    logging.info("onnx load model...")
    download_model(url)
    onnx_model = onnx.load(model_path)

    # inference
    logging.info("preprocessing...")
    input_ids, input_mask, segment_ids, extra_data, eval_examples = preprocess()

    m = None
    dev = device.create_cuda_gpu()
    n = len(input_ids)
    bs = batch_size
    all_results = []

    tmp_dict = {}
    for idx in range(0, n):
        logging.info("starting infer sample {}...".format(idx))
        item = eval_examples[idx]
        inputs = [
            np.array([item.qas_id], dtype=np.int32),
            segment_ids[idx:idx + bs].astype(np.int32),
            input_mask[idx:idx + bs].astype(np.int32),
            input_ids[idx:idx + bs].astype(np.int32),
        ]

        x_batch = []
        for inp in inputs:
            tmp_tensor = tensor.from_numpy(inp)
            tmp_tensor.to_device(dev)
            x_batch.append(tmp_tensor)

        # prepare the model
        if m is None:
            logging.info("model compling...")
            m = MyModel(onnx_model)
            # m.compile(x_batch, is_train=False, use_graph=True, sequential=True)

        logging.info("model running for sample {}...".format(idx))
        outputs = m.forward(*x_batch)

        logging.info("hanlde the result of sample {}...".format(idx))
        result = []
        for outp in outputs:
            result.append(tensor.to_numpy(outp))

        in_batch = result[1].shape[0]
        start_logits = [float(x) for x in result[1][0].flat]
        end_logits = [float(x) for x in result[0][0].flat]
        for i in range(0, in_batch):
            unique_id = len(all_results)
            all_results.append(
                RawResult(unique_id=unique_id,
                          start_logits=start_logits,
                          end_logits=end_logits))
    # postprocessing
    logging.info("postprocessing...")
    postprocess(eval_examples, extra_data, all_results)
