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
import configparser


def parse_config_arguments(config_path: str):
    parser = configparser.ConfigParser()
    parser.read(config_path)

    args = argparse.Namespace()

    # job config under DEFAULT
    args.log_name = parser.get('DEFAULT', 'log_name')
    args.budget = parser.getint('DEFAULT', 'budget')
    args.device = parser.get('DEFAULT', 'device')
    args.log_folder = parser.get('DEFAULT', 'log_folder')
    args.result_dir = parser.get('DEFAULT', 'result_dir')
    args.num_points = parser.getint('DEFAULT', 'num_points')
    args.max_load = parser.getint('DEFAULT', 'max_load')

    # sampler args
    args.search_space = parser.get('SAMPLER', 'search_space')
    args.population_size = parser.getint('SAMPLER', 'population_size')
    args.sample_size = parser.getint('SAMPLER', 'sample_size')
    args.simple_score_sum = parser.getboolean('SAMPLER', 'simple_score_sum')

    # nb101 args
    args.api_loc = parser.get('NB101', 'api_loc')
    args.init_channels = parser.getint('NB101', 'init_channels')
    args.bn = parser.getint('NB101', 'bn')
    args.num_stacks = parser.getint('NB101', 'num_stacks')
    args.num_modules_per_stack = parser.getint('NB101', 'num_modules_per_stack')

    # nb201 args
    args.init_w_type = parser.get('NB201', 'init_w_type')
    args.init_b_type = parser.get('NB201', 'init_b_type')
    args.arch_size = parser.getint('NB201', 'arch_size')

    # mlp args
    args.num_layers = parser.getint('MLP', 'num_layers')
    args.hidden_choice_len = parser.getint('MLP', 'hidden_choice_len')

    # mlp_trainer args
    args.epoch = parser.getint('MLP_TRAINER', 'epoch')
    args.batch_size = parser.getint('MLP_TRAINER', 'batch_size')
    args.lr = parser.getfloat('MLP_TRAINER', 'lr')
    args.patience = parser.getint('MLP_TRAINER', 'patience')
    args.iter_per_epoch = parser.getint('MLP_TRAINER', 'iter_per_epoch')
    args.nfeat = parser.getint('MLP_TRAINER', 'nfeat')
    args.nfield = parser.getint('MLP_TRAINER', 'nfield')
    args.nemb = parser.getint('MLP_TRAINER', 'nemb')
    args.report_freq = parser.getint('MLP_TRAINER', 'report_freq')
    args.workers = parser.getint('MLP_TRAINER', 'workers')

    # dataset args
    args.base_dir = parser.get('DATASET', 'base_dir')
    args.dataset = parser.get('DATASET', 'dataset')
    args.num_labels = parser.getint('DATASET', 'num_labels')

    # seq_train args
    args.worker_id = parser.getint('SEQ_TRAIN', 'worker_id')
    args.total_workers = parser.getint('SEQ_TRAIN', 'total_workers')
    args.total_models_per_worker = parser.getint('SEQ_TRAIN', 'total_models_per_worker')
    args.pre_partitioned_file = parser.get('SEQ_TRAIN', 'pre_partitioned_file')

    # dis_train args
    args.worker_each_gpu = parser.getint('DIS_TRAIN', 'worker_each_gpu')
    args.gpu_num = parser.getint('DIS_TRAIN', 'gpu_num')

    # tune_interval args
    args.kn_rate = parser.getint('TUNE_INTERVAL', 'kn_rate')

    # anytime args
    args.only_phase1 = parser.getboolean('ANYTIME', 'only_phase1')
    args.is_simulate = parser.getboolean('ANYTIME', 'is_simulate')

    # system performance exps
    args.models_explore = parser.getint('SYS_PERFORMANCE', 'models_explore')
    args.tfmem = parser.get('SYS_PERFORMANCE', 'tfmem')
    args.embedding_cache_filtering = parser.getboolean('SYS_PERFORMANCE', 'embedding_cache_filtering')
    args.concurrency = parser.getint('SYS_PERFORMANCE', 'concurrency')

    args.refinement_url = parser.get('SERVER', 'refinement_url')
    args.cache_svc_url = parser.get('SERVER', 'cache_svc_url')

    # db config
    args.db_name = parser.get('DB_CONFIG', 'db_name')
    args.db_user = parser.get('DB_CONFIG', 'db_user')
    args.db_host = parser.get('DB_CONFIG', 'db_host')
    args.db_port = parser.get('DB_CONFIG', 'db_port')

    return args