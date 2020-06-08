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
build_path = r'build/python'
sys.path.append(build_path)
model_path = r'examples/qabot'
sys.path.append(model_path)

from singa import autograd
from singa import layer
from singa import model
from singa import tensor
from singa import device
from singa import opt

from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import random

from data import parse_file, load_vocabulary, generate_qa_triplets, words_text_to_fixed_seqlen_vec, triplet_text_to_vec, train_data_gen_fn
from model import QAModel

# params
q_max_len=2
a_max_len=3
bs=32
embed_size=300
dev = device.create_cuda_gpu(set_default=False)

# embeding
embed_path = 'GoogleNews-vectors-negative300.bin'
wv = KeyedVectors.load_word2vec_format(embed_path, binary=True)
print("successfully loaded word2vec model")

# vocab
id_to_word, label_to_ans, label_to_ans_text = load_vocabulary('./V2/vocabulary', './V2/InsuranceQA.label2answer.token.encoded')
print("loaded vocab")

train_data = parse_file('./V2/InsuranceQA.question.anslabel.token.100.pool.solr.train.encoded', id_to_word, label_to_ans_text)
train_data = train_data[:100]
print("loaded train data")

train_triplets = generate_qa_triplets(train_data)

train_triplet_vecs = [triplet_text_to_vec(t, wv, q_max_len, a_max_len) for t in train_triplets]

train_data_gen = train_data_gen_fn(train_triplet_vecs, bs)

m = QAModel(3)
m.optimizer = opt.SGD()
tq = tensor.Tensor((bs, q_max_len, embed_size), dev, tensor.float32)
ta = tensor.Tensor((bs*2, a_max_len, embed_size), dev, tensor.float32)
tq.set_value(0.0)
ta.set_value(0.0)
m.compile([tq, ta], is_train=True, use_graph=False, sequential=False)

m.train()
for q, a in train_data_gen:
#     print(tq.shape)
#     print(ta.shape)
    tq.copy_from_numpy(q)
    ta.copy_from_numpy(a)
    score, l = m(tq,ta)
    print("l", l)