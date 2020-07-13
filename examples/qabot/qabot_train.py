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

import sys
build_path = r'build/python'
sys.path.append(build_path)
model_path = r'examples/qabot'
sys.path.append(model_path)

import time
from singa import model
from singa import tensor
from singa import device
from singa import opt

from gensim.models.keyedvectors import KeyedVectors
import numpy as np

from data import parse_file, parse_test_file, load_vocabulary, generate_qa_triplets, words_text_to_fixed_seqlen_vec, triplet_text_to_vec, train_data_gen_fn
from model import QAModel

# params
q_max_len = 15
a_max_len = 150
bs = 50 # as tq, ta use fix bs, bs should be factor of test size - 100
embed_size = 300
hidden_size = 100
max_epoch = 2

dev = device.create_cuda_gpu()
# dev = device.create_cuda_gpu_on(7)

# embeding
embed_path = 'GoogleNews-vectors-negative300.bin'
wv = KeyedVectors.load_word2vec_format(embed_path, binary=True)
print("successfully loaded word2vec model")

# vocab
id_to_word, label_to_ans, label_to_ans_text = load_vocabulary(
    './V2/vocabulary', './V2/InsuranceQA.label2answer.token.encoded')
print("loaded vocab")

train_data = parse_file(
    './V2/InsuranceQA.question.anslabel.token.100.pool.solr.train.encoded',
    id_to_word, label_to_ans_text)
test_data = parse_test_file(
    './V2/InsuranceQA.question.anslabel.token.100.pool.solr.test.encoded',
    id_to_word, label_to_ans_text)
# train_data = train_data[:100]
# test_data = test_data[:100]
print("loaded train data")


def load_model(max_bs, hidden_size):
    m = QAModel(hidden_size)
    m.optimizer = opt.SGD()
    tq = tensor.Tensor((max_bs, q_max_len, embed_size), dev, tensor.float32)
    ta = tensor.Tensor((max_bs * 2, a_max_len, embed_size), dev, tensor.float32)
    tq.set_value(0.0)
    ta.set_value(0.0)
    m.compile([tq, ta], is_train=True, use_graph=False, sequential=False)
    # m.compile([tq, ta], is_train=True, use_graph=True, sequential=True)
    # m.compile([tq, ta], is_train=True, use_graph=True, sequential=False)
    return m


def training_top1_hits(m, wv, q_max_len, a_max_len, train_data):
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


def training(m, all_train_data, max_epoch, eval_split_ratio=0.8):
    split_num = int(eval_split_ratio * len(all_train_data))
    train_data = all_train_data[:split_num]
    eval_data = all_train_data[split_num:]

    train_triplets = generate_qa_triplets(train_data) # triplet = <q, a+, a->
    train_triplet_vecs = [
        triplet_text_to_vec(t, wv, q_max_len, a_max_len) for t in train_triplets
    ] # triplet vecs = <q_vec, a+_vec, a-_vec>
    train_data_gen = train_data_gen_fn(train_triplet_vecs, bs)
    m.train()

    tq = tensor.Tensor((bs, q_max_len, embed_size), dev, tensor.float32)
    ta = tensor.Tensor((bs * 2, a_max_len, embed_size), dev, tensor.float32)
    for epoch in range(max_epoch):
        start = time.time()
        for q, a in train_data_gen: 
            #     print(tq.shape) # (bs,seq,embed)
            #     print(ta.shape) # (bs*2, seq, embed)
            tq.copy_from_numpy(q)
            ta.copy_from_numpy(a)
            score, l = m(tq, ta)
        top1hits = training_top1_hits(m, wv, q_max_len, a_max_len, train_data)
        print(
            "epoch %d, time used %d sec, top1 hits: %f, loss: " %
            (epoch, time.time() - start, top1hits), l)


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


def test_format(r, wv, q_max_len, a_max_len):
    q_text, labels, candis = r
    candis_vecs = [
        words_text_to_fixed_seqlen_vec(wv, label_to_ans_text[a_label],
                                       a_max_len) for a_label in candis
    ]
    if len(candis_vecs) % 2 == 1:
        candis_vecs.pop(-1)
    assert len(candis_vecs) % 2 == 0
    q_repeat = int(len(candis_vecs) / 2)
    q_vecs = [words_text_to_fixed_seqlen_vec(wv, q_text, q_max_len)] * q_repeat
    labels_idx = [candis.index(l) for l in labels if l in candis]
    return np.array(q_vecs), np.array(candis_vecs), labels, labels_idx

def testing(m, test_data):
    test_tuple_vecs = [
        test_format(r, wv, q_max_len, a_max_len) for r in test_data
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


m = load_model(bs, hidden_size)
training(m, train_data, max_epoch)
testing(m, test_data)
