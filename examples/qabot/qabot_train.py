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

import numpy as np
import time
import random
from tqdm import tqdm
import argparse

from singa import autograd, tensor, device, opt
from qabot_data import limit_encode_train, limit_encode_eval, prepare_data
from qabot_model import QAModel_maxpooling


def do_train(m, tq, ta, train, meta_data, args):
    '''
    batch size need to be large to see all negative ans
    '''
    m.train()
    for epoch in range(args.epochs):
        total_loss = 0
        start = time.time()

        q, ans_p, ans_n = limit_encode_train(train, meta_data['label2answer'],
                                             meta_data['idx2word'],
                                             args.q_seq_limit,
                                             args.ans_seq_limit,
                                             meta_data['idx2vec'])
        bs = args.bs

        for i in tqdm(range(len(q) // bs)):
            tq.copy_from_numpy(q[i * bs:(i + 1) * bs])
            a_batch = np.concatenate(
                [ans_p[i * bs:(i + 1) * bs], ans_n[i * bs:(i + 1) * bs]])
            ta.copy_from_numpy(a_batch)

            p_sim, n_sim = m.forward(tq, ta)
            l = autograd.ranking_loss(p_sim, n_sim)
            m.optimizer(l)

            total_loss += tensor.to_numpy(l)
        print(
            "epoch %d, time used %d sec, loss: " % (epoch, time.time() - start),
            total_loss * bs / len(q))


def do_eval(m, tq, ta, test, meta_data, args):
    q, candis, ans_count = limit_encode_eval(test, meta_data['label2answer'],
                                             meta_data['idx2word'],
                                             args.q_seq_limit,
                                             args.ans_seq_limit,
                                             meta_data['idx2vec'],
                                             args.number_of_candidates)
    m.eval()
    candi_pool_size = candis.shape[1]
    correct = 0
    start = time.time()
    for i in tqdm(range(len(q))):
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-m',
                        '--max-epoch',
                        default=30,
                        type=int,
                        help='maximum epochs',
                        dest='epochs')
    parser.add_argument('-b',
                        '--batch-size',
                        default=50,
                        type=int,
                        help='batch size',
                        dest='bs')
    parser.add_argument('-l',
                        '--learning-rate',
                        default=0.01,
                        type=float,
                        help='initial learning rate',
                        dest='lr')
    parser.add_argument('-i',
                        '--device-id',
                        default=0,
                        type=int,
                        help='which GPU to use',
                        dest='device_id')

    args = parser.parse_args()

    args.hid_s = 64
    args.q_seq_limit = 10
    args.ans_seq_limit = 50
    args.embed_size = 300
    args.number_of_candidates = args.bs * 2
    assert args.number_of_candidates <= 100, "number_of_candidates should be <= 100"

    dev = device.create_cuda_gpu_on(args.device_id)

    # tensor container
    tq = tensor.random((args.bs, args.q_seq_limit, args.embed_size), dev)
    ta = tensor.random((args.bs * 2, args.ans_seq_limit, args.embed_size), dev)

    # model
    m = QAModel_maxpooling(args.hid_s,
                           q_seq=args.q_seq_limit,
                           a_seq=args.ans_seq_limit)
    m.compile([tq, ta], is_train=True, use_graph=True, sequential=False)
    m.optimizer = opt.SGD(args.lr, 0.9)

    # get data
    train_raw, test_raw, label2answer, idx2word, idx2vec = prepare_data()
    meta_data = {
        'label2answer': label2answer,
        'idx2word': idx2word,
        'idx2vec': idx2vec
    }

    print("training...")
    do_train(m, tq, ta, train_raw, meta_data, args)

    print("Eval with train data...")
    do_eval(m, tq, ta, random.sample(train_raw, 2000), meta_data, args)

    print("Eval with test data...")
    do_eval(m, tq, ta, test_raw, meta_data, args)
