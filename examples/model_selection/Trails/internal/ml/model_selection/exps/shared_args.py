#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import os
import random
import numpy as np
import torch


def seed_everything(seed=2201):
    # 2022 -> 2021 -> 2031
    ''' [reference] https://gist.github.com/KirillVladimirov/005ec7f762293d2321385580d3dbe335 '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def sampler_args(parser):
    # define search space,
    parser.add_argument('--search_space', type=str, default="mlp_sp",
                        help='[nasbench101, nasbench201, mlp_sp]')
    # EA sampler's parameters,
    parser.add_argument('--population_size', type=int, default=10, help="The learning rate for REINFORCE.")
    parser.add_argument('--sample_size', type=int, default=3, help="The momentum value for EMA.")
    parser.add_argument('--simple_score_sum', default='True', type=str2bool,
                        help="Sum multiple TFMEM score or use Global Rank")


def space201_101_share_args(parser):
    parser.add_argument('--api_loc', type=str, default="NAS-Bench-201-v1_1-096897.pth",
                        help='which search space file to use, ['
                             'nasbench101: nasbench_only108.pkl'
                             'nasbench201: NAS-Bench-201-v1_1-096897.pth'
                             ' ... ]')

    parser.add_argument('--init_channels', default=16, type=int, help='output channels of stem convolution')
    parser.add_argument('--bn', type=int, default=1, help="If use batch norm in network 1 = true, 0 = false")


def nb101_args(parser):
    parser.add_argument('--num_stacks', default=3, type=int, help='#stacks of modules')
    parser.add_argument('--num_modules_per_stack', default=3, type=int, help='# modules per stack')


def nb201_args(parser):
    parser.add_argument('--init_w_type', type=str, default='none',
                        help='weight initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--init_b_type', type=str, default='none',
                        help='bias initialization (before pruning) type [none, xavier, kaiming, zero]')
    parser.add_argument('--arch_size', type=int, default=1,
                        help='How many node the architecture has at least')


def mlp_args(parser):
    parser.add_argument('--num_layers', default=4, type=int, help='# hidden layers')
    parser.add_argument('--hidden_choice_len', default=20, type=int, help=
                        'number of hidden layer choices, 10 for criteo, 20 for others')


def mlp_trainner_args(parser):
    parser.add_argument('--epoch', type=int, default=20,
                        help='number of maximum epochs, '
                             'frappe: 20, uci_diabetes: 40, criteo: 10'
                             'nb101: 108, nb201: 200')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help="learning reate")
    parser.add_argument('--patience', type=int, default=1, help='number of epochs for stopping training')
    # parser.add_argument('--eval_freq', type=int, default=10000, help='max number of batches to train per epoch')

    parser.add_argument('--iter_per_epoch', type=int, default=200,
                        help="None, "
                             "200 for frappe, uci_diabetes, "
                             "2000 for criteo")

    # MLP model config
    parser.add_argument('--nfeat', type=int, default=5500,
                        help='the number of features, '
                             'frappe: 5500, '
                             'uci_diabetes: 369,'
                             'criteo: 2100000')
    parser.add_argument('--nfield', type=int, default=10,
                        help='the number of fields, '
                             'frappe: 10, '
                             'uci_diabetes: 43,'
                             'criteo: 39')
    parser.add_argument('--nemb', type=int, default=10,
                        help='embedding size 10')

    # MLP train config
    parser.add_argument('--report_freq', type=int, default=30, help='report frequency')
    parser.add_argument('--workers', default=1, type=int, help='data loading workers')


def data_set_config(parser):
    parser.add_argument('--base_dir', type=str, default="./dataset/",
                        help='path of data and result parent folder')
    # define search space,
    parser.add_argument('--dataset', type=str, default='frappe',
                        help='cifar10, cifar100, ImageNet16-120 '
                             'frappe, criteo, uci_diabetes')

    parser.add_argument('--num_labels', type=int, default=2,
                        help='[10, 100, 120],'
                             '[2, 2, 2]')


def seq_train_all_params(parser):
    parser.add_argument('--worker_id', type=int, default=0, help='start from 0')
    parser.add_argument('--total_workers', type=int, default=120,
                        help='total number of workers, each train some models')
    parser.add_argument('--total_models_per_worker', type=int, default=-1, help='How many models to evaluate')
    parser.add_argument('--pre_partitioned_file',
                        default="./internal/ml/model_selection/exps/sampled_data/sampled_models_all.json",
                        type=str, help='all models with id')


def dis_train_all_models(parser):
    parser.add_argument('--worker_each_gpu', default=6, type=int, help='num worker each gpu')
    parser.add_argument('--gpu_num', default=8, type=int, help='num GPus')


# tune interval and schedule NK rate such that it can produce a good result
def tune_interval_NK_rate(parser):
    parser.add_argument('--kn_rate', default=-1, type=int, help="default N/K = 100")


def db4nas(parser):
    parser.add_argument('--db_name', default="pg_extension", type=str)
    parser.add_argument('--db_user', default="postgres", type=str)
    parser.add_argument('--db_host', default="localhost", type=str)
    parser.add_argument('--db_port', default=28814, type=int)


def anytime_exp_set(parser):
    parser.add_argument('--only_phase1', default='False', type=str2bool)
    parser.add_argument('--is_simulate', default='True', type=str2bool,
                        help='Use pre-computed result or run online')


def system_performance_exp(parser):
    parser.add_argument('--models_explore', default=10, type=int, help='# models to explore in the filtering phase')
    parser.add_argument('--tfmem', default="jacflow", type=str, help='the matrix t use, all_matrix')
    parser.add_argument('--embedding_cache_filtering', default='True', type=str2bool,
                        help='Cache embedding for MLP in filtering phase?')
    parser.add_argument('--concurrency', default=1, type=int, help='number of worker in filtering phase')


def parse_arguments():
    parser = argparse.ArgumentParser(description='system')

    # job config
    parser.add_argument('--log_name', type=str, default="main_T_100s")
    parser.add_argument('--budget', type=int, default=100, help="in second")

    # define base dir, where it stores apis, datasets, logs, etc,
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--local_rank', type=int, default=1, help="local rank")

    parser.add_argument('--log_folder', default="log_debug", type=str)

    parser.add_argument('--result_dir', default="./internal/ml/model_selection/exp_result/", type=str,
                        help='path to store exp outputs')
    parser.add_argument('--num_points', default=12, type=int, help='num GPus')

    sampler_args(parser)

    nb101_args(parser)
    nb201_args(parser)
    space201_101_share_args(parser)

    mlp_args(parser)
    data_set_config(parser)
    mlp_trainner_args(parser)
    seq_train_all_params(parser)
    dis_train_all_models(parser)

    tune_interval_NK_rate(parser)

    db4nas(parser)
    anytime_exp_set(parser)

    system_performance_exp(parser)

    # tmp
    parser.add_argument('--max_load', type=int, default=-1, help="Max Loading time")

    # refinement server
    parser.add_argument('--url', type=str, default=-1, help="Max Loading time")

    seed_everything()

    return parser.parse_args()
