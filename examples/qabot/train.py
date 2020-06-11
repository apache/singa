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
bs = 32  # as tq, ta use fix bs, bs should be factor of test size - 100


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
    return word_to_vec, label_to_ans, train_raw_data, test_raw_data


def load_model(max_bs, hidden_size):
    m = QAModel(hidden_size)
    m.optimizer = opt.SGD(lr=0.005, momentum=0.9, weight_decay=1e-5)
    tq = tensor.Tensor((max_bs, q_max_len, embed_size), dev, tensor.float32)
    ta = tensor.Tensor((max_bs * 2, a_max_len, embed_size), dev, tensor.float32)
    tq.set_value(0.0)
    ta.set_value(0.0)
    m.compile([tq, ta], is_train=True, use_graph=False, sequential=False)
    # m.compile([tq, ta], is_train=True, use_graph=True, sequential=True)
    # m.compile([tq, ta], is_train=True, use_graph=True, sequential=False)
    return m


def training_top1_hits(m, dev, wv, q_max_len, a_max_len, train_data):
    m.eval()
    hits = 0
    train_eval_data = [
        train_eval_format(r, wv, q_max_len, a_max_len) for r in train_data
    ]
    trials = len(train_eval_data)
    for q, a in train_eval_data:
        tq = tensor.from_numpy(q.astype(np.float32))
        ta = tensor.from_numpy(a.astype(np.float32))
        tq.to_device(dev)
        ta.to_device(dev)
        out = m.forward(tq, ta)
        sim_first_half = tensor.to_numpy(out[0])
        sim_second_half = tensor.to_numpy(out[1])
        sim = np.concatenate([sim_first_half, sim_second_half]).flatten()
        if np.argmax(sim) == 0:
            hits += 1
    # print("training top1 hits rate: ", hits/trials)
    return hits / trials


def training(m, dev, wv, all_train_data, max_epoch, eval_split_ratio=0.8):
    split_num = int(eval_split_ratio * len(all_train_data))
    train_data = all_train_data[:split_num]
    eval_data = all_train_data[split_num:]
    train_data = train_data[:10]
    eval_data = eval_data[:10]

    train_triplets = generate_qa_triplets(train_data)  # triplet = <q, a+, a->
    train_triplet_vecs = [
        triplet_text_to_vec(t, wv, q_max_len, a_max_len) for t in train_triplets
    ]  # triplet vecs = <q_vec, a+_vec, a-_vec>

    tq = tensor.Tensor((bs, q_max_len, embed_size), dev, tensor.float32)
    ta = tensor.Tensor((bs * 2, a_max_len, embed_size), dev, tensor.float32)

    for epoch in range(max_epoch):
        start = time.time()

        train_loss = 0
        train_data_gen = train_data_gen_fn(train_triplet_vecs, bs)

        m.train()
        for q, a in train_data_gen:
            #     print(tq.shape) # (bs,seq,embed)
            #     print(ta.shape) # (bs*2, seq, embed)
            tq.copy_from_numpy(q)
            ta.copy_from_numpy(a)
            score, l = m(tq, ta)
            train_loss += l

        top1hits = training_top1_hits(m, dev, wv, q_max_len, a_max_len,
                                      train_data)
        print(
            "epoch %d, time used %d sec, top1 hits: %f, loss: " %
            (epoch, time.time() - start, top1hits), train_loss)


def train_eval_format(row, wv, q_max_len, a_max_len):
    q, apos, anegs = row
    all_a = [apos] + anegs
    a_vecs = [words_text_to_fixed_seqlen_vec(wv, a, a_max_len) for a in all_a]
    if len(a_vecs) % 2 == 1:
        a_vecs.pop(-1)
    assert len(a_vecs) % 2 == 0
    q_repeat = int(len(a_vecs) / 2)
    q_vecs = [words_text_to_fixed_seqlen_vec(wv, q, q_max_len)] * q_repeat
    return np.array(q_vecs), np.array(a_vecs)


def test_format(r, wv, label_to_ans, q_max_len, a_max_len):
    q_text, labels, candis = r
    candis_vecs = [
        words_text_to_fixed_seqlen_vec(wv, label_to_ans[a_label], a_max_len)
        for a_label in candis
    ]
    if len(candis_vecs) % 2 == 1:
        candis_vecs.pop(-1)
    assert len(candis_vecs) % 2 == 0
    q_repeat = int(len(candis_vecs) / 2)
    q_vecs = [words_text_to_fixed_seqlen_vec(wv, q_text, q_max_len)] * q_repeat
    labels_idx = [candis.index(l) for l in labels if l in candis]
    return np.array(q_vecs), np.array(candis_vecs), labels, labels_idx


def testing(m, dev, wv, label_to_ans, test_data):
    test_tuple_vecs = [
        test_format(r, wv, label_to_ans, q_max_len, a_max_len)
        for r in test_data
    ]
    m.eval()
    hits = 0
    trials = len(test_tuple_vecs)

    tq = tensor.Tensor((bs, q_max_len, embed_size), dev, tensor.float32)
    ta = tensor.Tensor((bs * 2, a_max_len, embed_size), dev, tensor.float32)
    for q, a, labels, labels_idx in test_tuple_vecs:
        # print(q.shape) # (50, seq, embed)
        # print(a.shape) # (100, seq, embed)
        tq.copy_from_numpy(q)
        ta.copy_from_numpy(a)
        out = m.forward(tq, ta)
        sim_first_half = tensor.to_numpy(out[0])
        sim_second_half = tensor.to_numpy(out[1])
        sim = np.concatenate([sim_first_half, sim_second_half]).flatten()
        if np.argmax(sim) in labels_idx:
            hits += 1
    print("training top1 hits rate: ", hits / trials)


def run(device_id, max_epoch, batch_size, sgd, graph, verbosity):
    # 1. create device
    dev = device.create_cuda_gpu_on(device_id)
    dev.SetVerbosity(verbosity)
    dev.SetRandSeed(0)
    np.random.seed(0)

    # 2. load data
    word_to_vec, label_to_ans, train_data, test_data = load_data()

    # 3. create placeholders
    tq = tensor.Tensor((batch_size, q_max_len, embed_size), dev)
    ta = tensor.Tensor((batch_size * 2, a_max_len, embed_size), dev)

    # 4. load model
    model = QAModel(hidden_size)
    model.set_optimizer(sgd)
    model.compile([tq, ta], is_train=True, use_graph=graph, sequential=False)

    # 5. training
    training(model, dev, word_to_vec, train_data, max_epoch)

    # 6. testing
    testing(model, dev, word_to_vec, label_to_ans, test_data)


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
