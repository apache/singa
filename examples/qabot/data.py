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

from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import random


def load_vocabulary(vocab_path, label_path):
    id_to_word = {}
    with open(vocab_path, 'rb') as f:
        lines = f.readlines()
        for l in lines:
            d = l.rstrip().decode("utf-8").split("\t")
            if d[0] not in id_to_word:
                id_to_word[d[0]] = d[1]

    label_to_ans = {}
    label_to_ans_text = {}
    with open(label_path) as f:
        lines = f.readlines()
        for l in lines:
            label, answer = l.rstrip().split('\t')
            if label not in label_to_ans:
                label_to_ans[label] = answer
                label_to_ans_text[label] = [
                    id_to_word[t] for t in answer.split(' ')
                ]
    return id_to_word, label_to_ans, label_to_ans_text


def parse_file(fpath, id_to_word, label_to_ans_text):
    data = []
    with open(fpath) as f:
        lines = f.readlines()
        for l in lines:
            d = l.rstrip().split('\t')
            q = [id_to_word[t] for t in d[1].split(' ')]  # question
            poss = [label_to_ans_text[t] for t in d[2].split(' ')
                   ]  # ground-truth
            negs = [
                label_to_ans_text[t] for t in d[3].split(' ') if t not in d[2]
            ]  # candidate-pool without ground-truth
            for pos in poss:
                data.append((q, pos, negs))
    return data


def parse_test_file(fpath, id_to_word, label_to_ans_text):
    data = []
    with open(fpath) as f:
        lines = f.readlines()
        for l in lines[12:]:
            d = l.rstrip().split('\t')
            q = [id_to_word[t] for t in d[1].split(' ')]  # question
            poss = [t for t in d[2].split(' ')]  # ground-truth
            cands = [t for t in d[3].split(' ')]  # candidate-pool
            data.append((q, poss, cands))
    return data

def words_text_to_fixed_seqlen_vec(word2vec, words, sentence_length=10):
    sentence_vec = []
    for word in words:
        if len(sentence_vec) >= sentence_length:
            break
        if word in word2vec:
            sentence_vec.append(word2vec[word])
        else:
            sentence_vec.append(np.zeros((300,)))
    while len(sentence_vec) < sentence_length:
        sentence_vec.append(np.zeros((300,)))
    return np.array(sentence_vec, dtype=np.float32)


def generate_qa_triplets(data, num_negs=10):
    tuples = []
    for (q, a_pos, a_negs) in data:
        for i in range(num_negs):
            tpl = (q, a_pos, random.choice(a_negs))
            tuples.append(tpl)
    return tuples

def qa_tuples_to_naive_training_format(wv, tuples):
    training = []
    q_len_limit = 10
    a_len_limit = 100
    for tpl in tuples:
        q, a_pos, a_neg = tpl
        q_vec = words_text_to_fixed_seqlen_vec(wv, q, q_len_limit)
        training.append(
            (q_vec, words_text_to_fixed_seqlen_vec(wv, a_pos, a_len_limit), 1))
        training.append(
            (q_vec, words_text_to_fixed_seqlen_vec(wv, a_neg, a_len_limit), 0))
    return training

def triplet_text_to_vec(triplet, wv, q_max_len, a_max_len):
    return [
        words_text_to_fixed_seqlen_vec(wv, triplet[0], q_max_len),
        words_text_to_fixed_seqlen_vec(wv, triplet[1], a_max_len),
        words_text_to_fixed_seqlen_vec(wv, triplet[2], a_max_len)
    ]

def train_data_gen_fn(train_triplet_vecs, bs=32):
    q = []
    ap = []
    an = []
    for t in train_triplet_vecs:
        q.append(t[0])
        ap.append(t[1])
        an.append(t[2])
        if len(q) >= bs:
            q = np.array(q)
            ap = np.array(ap)
            an = np.array(an)
            a = np.concatenate([ap, an])
            assert 2 * q.shape[0] == a.shape[0]
            yield q, a
            q = []
            ap = []
            an = []
    # return the rest
    # return np.array(q), np.concatenate([np.array(ap), np.array(an)])


if __name__ == "__main__":
    pass
