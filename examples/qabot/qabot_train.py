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
import numpy as np
import pickle
import time
build_path = r'build/python'
sys.path.append(build_path)
model_path = r'examples/qabot'
sys.path.append(model_path)

from singa import autograd, layer, model, tensor, device, opt
from qabot_data import *
from qabot_model import *


# questions_encoded, poss_encoded, negs_encoded
def do_train(m, tq, ta, q_seq_limit, ans_seq_limit, bs, max_epoch=30):
    '''
    need to be large to see all negative ans
    '''
    train_raw, test_raw, label2answer, idx2word, idx2vec = prepare_data()
    m.train()
    for epoch in range(max_epoch):
        total_loss = 0
        start = time.time()

        q, ans_p, ans_n = limit_encode_train(train_raw, label2answer, idx2word, q_seq_limit, ans_seq_limit, idx2vec)

        for i in range(len(q) // bs):
            tq.copy_from_numpy(q[i * bs:(i + 1) * bs])
            ta.copy_from_numpy(
                np.concatenate(
                    [ans_p[i * bs:(i + 1) * bs], ans_n[i * bs:(i + 1) * bs]]))

            out = m.forward(tq, ta)
            l = autograd.qa_lstm_loss(out[0], out[1])
            m.optimizer(l)

            total_loss += tensor.to_numpy(l)
        if epoch % 1 == 0:
            print(
                "epoch %d, time used %d sec, loss: " %
                (epoch, time.time() - start), total_loss * bs / len(q))


def do_eval(m, tq, ta, q, candis, ans_count):
    m.eval()
    candi_pool_size = candis.shape[1]
    correct = 0
    start = time.time()
    for i in range(len(q)):
        # batch size bs must satisfy: bs == repeated q, bs == number of answers//2
        # 1 question repeat n times, n == number of answers//2
        _q = np.repeat([q[i]], candi_pool_size // 2, axis=0)
        tq.copy_from_numpy(_q)
        ta.copy_from_numpy(candis[i])

        (first_half_score, second_half_score) = m.forward(tq, ta)

        first_half_score = tensor.to_numpy(first_half_score)
        second_half_score = tensor.to_numpy(second_half_score)
        scores = np.concatenate((first_half_score, second_half_score))
        pred_max_idx = np.argmax(scores)

        if pred_max_idx < ans_count[i]:
            correct += 1

    print("eval top %s " % (candi_pool_size), " accuracy", correct / len(q),
          " time used %d sec" % (time.time() - start))


if __name__ == "__main__":
    # parameters
    hidden_size = 64
    bs = 50
    q_seq_limit = 10
    ans_seq_limit = 30
    embed_size = 300
    number_of_candidates = bs * 2
    epochs = 30

    # set up model
    dev = device.create_cuda_gpu_on(7)
    tq = tensor.random((bs, q_seq_limit, embed_size), dev)
    ta = tensor.random((bs * 2, ans_seq_limit, embed_size), dev)
    m = QAModelv2(hidden_size, q_seq=q_seq_limit, a_seq=ans_seq_limit)

    # get data
    train_raw, test_raw, label2answer, idx2word, idx2vec = prepare_data()
    # train_raw = train_raw[:1000]
    # test_raw = test_raw[:100]

    # train
    train_questions_encoded, train_poss_encoded, train_negs_encoded = limit_encode_train(
        train_raw, label2answer, idx2word, q_seq_limit, ans_seq_limit, idx2vec)
    do_train(m, tq, ta, train_questions_encoded, train_poss_encoded,
        train_negs_encoded, epochs)

    # eval on train data
    train_raw = train_raw[:1000]
    eval_questions_encoded, eval_candi_pools_encoded, eval_ans_count = limit_encode_eval(
        train_raw, label2answer, idx2word, q_seq_limit, ans_seq_limit, idx2vec,
        number_of_candidates)
    do_eval(m, tq, ta, eval_questions_encoded, eval_candi_pools_encoded,
            eval_ans_count)

    # eval on test data
    test_questions_encoded, test_candi_pools_encoded, test_ans_count = limit_encode_eval(
        test_raw, label2answer, idx2word, q_seq_limit, ans_seq_limit, idx2vec,
        number_of_candidates)
    do_eval(m, tq, ta, test_questions_encoded, test_candi_pools_encoded,
            test_ans_count)