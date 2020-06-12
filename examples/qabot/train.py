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

import time
import random

import argparse
import numpy as np
import gensim.models.keyedvectors as kv

from singa import device
from singa import tensor
from singa import opt

from data import *
from model import QAModel

q_max_len = 15
a_max_len = 150
embed_size = 300
hidden_size = 100


def load_data():
    # file paths
    embed_path = 'GoogleNews-vectors-negative300.bin'
    vocab_path = 'V2/vocabulary'
    label_path = 'V2/InsuranceQA.label2answer.token.encoded'
    train_path = 'V2/InsuranceQA.question.anslabel.token.100.pool.solr.train.encoded'
    test_path = 'V2/InsuranceQA.question.anslabel.token.100.pool.solr.test.encoded'

    # load word2vec model and corpus
    word_to_vec = kv.KeyedVectors.load_word2vec_format(embed_path, binary=True)
    id_to_word, label_to_ans = load_vocabulary(vocab_path, label_path)
    train_raw_data = parse_train_file(train_path, id_to_word, label_to_ans)
    test_raw_data = parse_test_file(test_path, id_to_word, label_to_ans)
    print('successfully loaded word2vec model and corpus')

    # cut part of the data
    train_raw_data = train_raw_data[:200]
    test_raw_data = test_raw_data[:200]

    # process raw data
    train_data = generate_train_data(train_raw_data, word_to_vec)
    eval_data = generate_eval_data(train_raw_data, word_to_vec)
    test_data = generate_test_data(test_raw_data, word_to_vec, label_to_ans)
    print('successfully generated train, eval, test data')

    return train_data, eval_data, test_data


def run(device_id, max_epoch, batch_size, sgd, graph, verbosity):
    # 1. create device
    dev = device.create_cuda_gpu_on(device_id)
    dev.SetVerbosity(verbosity)
    dev.SetRandSeed(0)
    np.random.seed(0)

    # 2. load data
    train_data, eval_data, test_data = load_data()
    num_train_batch = train_data[0].shape[0] // batch_size
    num_eval_batch = len(eval_data)
    num_test_batch = len(test_data)

    # 3. create placeholders
    tq = tensor.Tensor((batch_size, q_max_len, embed_size), dev)
    ta = tensor.Tensor((batch_size * 2, a_max_len, embed_size), dev)

    # 4. load model
    model = QAModel(hidden_size)
    model.set_optimizer(sgd)
    model.compile([tq, ta], is_train=True, use_graph=graph, sequential=False)

    # 5. training
    for epoch in range(max_epoch):
        start = time.time()

        # 5-1 train the model
        train_loss = 0
        model.train()
        for b in range(num_train_batch):
            q = train_data[0][b * batch_size:(b + 1) * batch_size]
            a = train_data[1][b * batch_size * 2:(b + 1) * batch_size * 2]
            tq.copy_from_numpy(q)
            ta.copy_from_numpy(a)
            score, loss = model(tq, ta)
            train_loss += loss

        # 5-2 evaluate the model
        hits = 0
        model.eval()
        for b in range(num_eval_batch):
            q, a = eval_data[b]
            q = tensor.from_numpy(q.astype(np.float32))
            a = tensor.from_numpy(a.astype(np.float32))
            q.to_device(dev)
            a.to_device(dev)
            out = model.forward(q, a)

            sim_first_half = tensor.to_numpy(out[0])
            sim_second_half = tensor.to_numpy(out[1])
            sim = np.concatenate([sim_first_half, sim_second_half]).flatten()
            if np.argmax(sim) == 0:
                hits += 1

        top1hits = hits / num_eval_batch
        elapsed_time = time.time() - start
        print(
            "epoch %d, time used %d sec, top1 hits: %f, loss: " %
            (epoch, elapsed_time, top1hits), train_loss)

    # 6. testing
    batch_size = 100 / 2  # 100: the number of candidate answers
    tq = tensor.Tensor((50, q_max_len, embed_size), dev)
    ta = tensor.Tensor((50 * 2, a_max_len, embed_size), dev)

    hits = 0
    model.eval()
    for b in range(num_test_batch):
        q, a, labels, labels_idx = test_data[b]
        tq.copy_from_numpy(q)
        ta.copy_from_numpy(a)
        out = model.forward(tq, ta)
        sim_first_half = tensor.to_numpy(out[0])
        sim_second_half = tensor.to_numpy(out[1])
        sim = np.concatenate([sim_first_half, sim_second_half]).flatten()
        if np.argmax(sim) == 0:
            hits += 1
    print("training top1 hits rate: ", hits / num_test_batch)

    dev.PrintTimeProfiling()


if __name__ == '__main__':
    # use argparse to get command config: max_epoch, batch_size, etc. for single gpu training
    parser = argparse.ArgumentParser(
        description='Training using the autograd and graph.')
    parser.add_argument('-m',
                        '--max-epoch',
                        default=10,
                        type=int,
                        help='maximum epochs',
                        dest='max_epoch')
    parser.add_argument('-b',
                        '--batch-size',
                        default=64,
                        type=int,
                        help='batch size',
                        dest='batch_size')
    parser.add_argument('-l',
                        '--learning-rate',
                        default=0.005,
                        type=float,
                        help='initial learning rate',
                        dest='lr')
    # determine which gpu to use
    parser.add_argument('-i',
                        '--device-id',
                        default=0,
                        type=int,
                        help='which GPU to use',
                        dest='device_id')
    parser.add_argument('-g',
                        '--disable-graph',
                        default='True',
                        action='store_false',
                        help='disable graph',
                        dest='graph')
    parser.add_argument('-v',
                        '--log-verbosity',
                        default=0,
                        type=int,
                        help='logging verbosity',
                        dest='verbosity')

    args = parser.parse_args()

    sgd = opt.SGD(lr=args.lr, momentum=0.9, weight_decay=1e-5)
    run(args.device_id, args.max_epoch, args.batch_size, sgd, args.graph,
        args.verbosity)
