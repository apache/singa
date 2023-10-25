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

# frappe
python exps/main_v2/ground_truth/2.seq_train_dist_online.py  \
    --log_name=baseline_train_based \
    --search_space=mlp_sp \
    --num_layers=4 \
    --hidden_choice_len=20 \
    --base_dir=../exp_data/ \
    --num_labels=1 \
    --device=gpu \
    --batch_size=1024 \
    --lr=0.001 \
    --epoch=10 \
    --iter_per_epoch=100 \
    --dataset=frappe \
    --nfeat=5500 \
    --nfield=10 \
    --nemb=10 \
    --total_models_per_worker=10 \
    --workers=0 \
    --worker_each_gpu=1 \
    --gpu_num=8 \
    --log_folder=LogFrappee \
    --pre_partitioned_file=./exps/main_v2/ground_truth/sampled_models_10000_models.json &

# criteo
python exps/main_v2/ground_truth/2.seq_train_dist_online.py  \
    --log_name=baseline_train_based \
    --search_space=mlp_sp \
    --num_layers=4 \
    --hidden_choice_len=10 \
    --base_dir=../exp_data/ \
    --num_labels=1 \
    --device=gpu \
    --batch_size=1024 \
    --lr=0.001 \
    --epoch=10 \
    --iter_per_epoch=2000 \
    --dataset=criteo \
    --nfeat=2100000 \
    --nfield=39 \
    --nemb=10 \
    --workers=0 \
    --worker_each_gpu=9 \
    --gpu_num=8 \
    --log_folder=LogCriteo \
    --pre_partitioned_file=./exps/main_v2/ground_truth/sampled_models_10000_models.json &
