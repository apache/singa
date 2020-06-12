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


# auxiliary function
def split_one_line(line, decode=False):
    if decode:
        return line.rstrip().decode("utf-8").split("\t")
    else:
        return line.rstrip().split("\t")


def words_to_fixed_length_vec(word_to_vec, words, sentence_length=10):
    sentence_vec = []
    for word in words:
        if len(sentence_vec) >= sentence_length:
            break
        if word in word_to_vec:
            sentence_vec.append(word_to_vec[word])
        else:
            sentence_vec.append(np.zeros((300,)))
    while len(sentence_vec) < sentence_length:
        sentence_vec.append(np.zeros((300,)))
    return np.array(sentence_vec, dtype=np.float32)


# functions for parsing files
def load_vocabulary(vocab_path, label_path):
    id_to_word = {}
    with open(vocab_path, 'rb') as f:
        lines = f.readlines()
        for line in lines:
            word_id, word = split_one_line(line, decode=True)
            if word_id not in id_to_word:
                id_to_word[word_id] = word

    label_to_ans = {}
    with open(label_path) as f:
        lines = f.readlines()
        for line in lines:
            label, answer = split_one_line(line)
            if label not in label_to_ans:
                ans_text_list = [id_to_word[t] for t in answer.split(' ')]
                label_to_ans[label] = ans_text_list

    return id_to_word, label_to_ans


def parse_train_file(fpath, id_to_word, label_to_ans):
    data = []
    with open(fpath) as f:
        lines = f.readlines()
        for line in lines:
            # domain, question, ground-truth, candidate-pool without ground-truth
            domain, question, poss, negs = split_one_line(line)
            question = [id_to_word[t] for t in question.split(' ')]
            poss = [label_to_ans[t] for t in poss.split(' ')]
            negs = [label_to_ans[t] for t in negs.split(' ') if t not in poss]

            for pos in poss:
                data.append((question, pos, negs))

    return data


def parse_test_file(fpath, id_to_word, label_to_ans):
    data = []
    with open(fpath) as f:
        lines = f.readlines()
        for line in lines[12:]:
            # domain, question, ground-truth, candidate-pool
            domain, question, poss, cands = split_one_line(line)
            question = [id_to_word[t] for t in question.split(' ')]
            poss = [t for t in poss.split(' ')]
            cands = [t for t in cands.split(' ')]

            data.append((question, poss, cands))

    return data


# functions for generating data
def generate_train_data(train_raw_data, word_to_vec, num_negs=10):
    train_q = []
    train_a_pos = []
    train_a_neg = []
    q_max_len = 15
    a_max_len = 150
    for (question, a_pos, a_negs) in train_raw_data:
        for i in range(num_negs):
            a_neg = random.choice(a_negs)
            q_vec = words_to_fixed_length_vec(word_to_vec, question, q_max_len)
            a_pos_vec = words_to_fixed_length_vec(word_to_vec, a_pos, a_max_len)
            a_neg_vec = words_to_fixed_length_vec(word_to_vec, a_neg, a_max_len)
            train_q.append(np.array(q_vec))
            train_a_pos.append(np.array(a_pos_vec))
            train_a_neg.append(np.array(a_neg_vec))

    train_q = np.array(train_q)
    train_a_pos = np.array(train_a_pos)
    train_a_neg = np.array(train_a_neg)
    train_a = np.concatenate([train_a_pos, train_a_neg])
    return (train_q, train_a)


def generate_eval_data(train_raw_data, word_to_vec):
    eval_data = []
    q_max_len = 15
    a_max_len = 150
    for (question, a_pos, a_negs) in train_raw_data:
        a_vecs = []
        for ans in [a_pos] + a_negs:
            a_vec = words_to_fixed_length_vec(word_to_vec, ans, a_max_len)
            a_vecs.append(np.array(a_vec))
        if len(a_vecs) % 2 == 1:
            a_vecs.pop(-1)
        assert len(a_vecs) % 2 == 0

        repeat_times = int(len(a_vecs) / 2)
        q_vec = words_to_fixed_length_vec(word_to_vec, question, q_max_len)
        q_vecs = [q_vec] * repeat_times

        q_vecs = np.array(q_vecs)
        a_vecs = np.array(a_vecs)
        eval_data.append((q_vecs, a_vecs))

    return eval_data


def generate_test_data(train_test_data, word_to_vec, label_to_ans):
    test_data = []
    q_max_len = 15
    a_max_len = 150
    for (question, labels, cands) in train_test_data:
        cands_vecs = []
        for cand in cands:
            text = label_to_ans[cand]
            cand_vec = words_to_fixed_length_vec(word_to_vec, text, a_max_len)
            cands_vecs.append(cand_vec)
        if len(cands_vecs) % 2 == 1:
            cands_vecs.pop(-1)
        assert len(cands_vecs) % 2 == 0

        repeat_times = int(len(cands_vecs) / 2)
        q_vec = words_to_fixed_length_vec(word_to_vec, question, q_max_len)
        q_vecs = [q_vec] * repeat_times

        labels_idx = [cands.index(label) for label in labels if label in cands]

        q_vecs = np.array(q_vecs)
        cands_vecs = np.array(cands_vecs)
        test_data.append((q_vecs, cands_vecs, labels, labels_idx))

    return test_data


if __name__ == "__main__":
    pass
