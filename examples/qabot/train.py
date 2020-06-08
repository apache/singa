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
import sys
new_path = r'/root/singa/build/python'
sys.path.append(new_path)

from data import *
from model import QAModel, MLP #, QALoss

from singa import autograd
from singa import layer
from singa import model
from singa import tensor
from singa import device
from singa import opt
from tqdm import tqdm

import singa.singa_wrap as singa

import numpy as np

def train(m, tq, ta, id_to_word, label_to_ans_text, wv):
    print("training")
    train_data = parse_file('./V2/InsuranceQA.question.anslabel.token.100.pool.solr.train.encoded', id_to_word, label_to_ans_text)
    train_data = train_data[:100]
    train_triplets = generate_qa_triplets(train_data) # (q, a+, a-)


    for epoch in range(max_epoch):
        for (q,apos,aneg) in tqdm(train_triplets):
            q = words_text_to_fixed_seqlen_vec(wv,q,q_seq_length)
            a = np.array([words_text_to_fixed_seqlen_vec(wv,apos,a_seq_length),words_text_to_fixed_seqlen_vec(wv,aneg,a_seq_length)])
            q = q.astype(np.float32)
            a = a.astype(np.float32)

            tq.copy_from_numpy(q)
            ta.copy_from_numpy(a)

            # train
            _, l = m(tq,ta)

        print("loss", l)

        # training top1 accuracy
        top1hit = 0
        trials = len(train_data)

        for (q, a_pos, a_negs) in train_data:
            scores = []
            q_vec = words_text_to_fixed_seqlen_vec(wv,q,q_seq_length)
            tq.copy_from_numpy(q_vec)

            a_pos_vec = words_text_to_fixed_seqlen_vec(wv,a_pos,a_seq_length)
            # prepare for <q, a+, a+> input
            ta.copy_from_numpy(np.array([a_pos_vec]*2))
            true_score, l = m(tq,ta)

            a_neg_vecs = [words_text_to_fixed_seqlen_vec(wv,a_neg,a_seq_length) for a_neg in a_negs]

            # prepare for triplets <q, a-, a-> input
            while len(a_neg_vecs) > 1:
                a_vec=[]
                a_vec.append(a_neg_vecs.pop(0))
                a_vec.append(a_neg_vecs.pop(0))
                ta.copy_from_numpy(np.array(a_vec))
                score, l = m(tq,ta)
                scores.extend(score)

            max_neg = np.max(np.array([tensor.to_numpy(s) for s in scores]).flatten())
            if max_neg < tensor.to_numpy(true_score[0])[0]:
                top1hit+=1

        print("training top 1 hit accuracy: ", top1hit/trials)


def test(m, tq, ta, id_to_word, label_to_ans_text, wv):
    print("testing")
    test_data = parse_test_file('./V2/InsuranceQA.question.anslabel.token.100.pool.solr.test.encoded', id_to_word, label_to_ans_text)
    test_data = test_data[:10]  # run on n samples

    m.eval()
    top1hit=0
    trials = len(test_data)
    for (q, labels, cands) in test_data:

        q_vec = words_text_to_fixed_seqlen_vec(wv, q, q_seq_length)
        tq.copy_from_numpy(np.array(q_vec))

        cands_vec = [words_text_to_fixed_seqlen_vec(wv, label_to_ans_text[candidate_label], a_seq_length) for candidate_label in cands]

        scores = []
        # inference all candidates
        # import pdb; pdb.set_trace()
        while len(cands_vec) > 1:
            a_vec=[]
            a_vec.append(cands_vec.pop(0))
            a_vec.append(cands_vec.pop(0))
            ta.copy_from_numpy(np.array(a_vec))
            score = m(tq,ta) # inference mode only return forward result
            scores.extend(score)

        # check correct from predict
        true_idxs = [cands.index(l) for l in labels if l in cands]
        pred_idx = np.argmax(np.array([tensor.to_numpy(s) for s in scores]).flatten())
        if pred_idx in true_idxs:
            top1hit += 1

    print("testing top 1 hit accuracy: ", top1hit/trials)

if __name__ == "__main__":
    dev = device.create_cuda_gpu(set_default=False)

    q_seq_length = 10
    a_seq_length = 100
    embed_size = 300
    batch_size = 128
    max_epoch = 30
    hidden_size = 100

    # build model
    m = QAModel(hidden_size)
    print("created qa model")
    # m = MLP()
    m.optimizer = opt.SGD()

    tq = tensor.Tensor((1, q_seq_length, embed_size), dev, tensor.float32)
    ta = tensor.Tensor((2, a_seq_length, embed_size), dev, tensor.float32)

    tq.set_value(0.0)
    ta.set_value(0.0)

    m.compile([tq, ta], is_train=True, use_graph=False, sequential=False)

    # embeding
    embed_path = 'GoogleNews-vectors-negative300.bin'
    wv = KeyedVectors.load_word2vec_format(embed_path, binary=True)
    print("successfully loaded word2vec model")

    # vocab data
    id_to_word, label_to_ans, label_to_ans_text = load_vocabulary('./V2/vocabulary', './V2/InsuranceQA.label2answer.token.encoded')

    train(m, tq, ta, id_to_word, label_to_ans_text, wv)

    test(m, tq, ta, id_to_word, label_to_ans_text, wv)
