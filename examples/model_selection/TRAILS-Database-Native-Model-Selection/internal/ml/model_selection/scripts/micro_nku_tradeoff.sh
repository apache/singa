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

export PYTHONPATH=$PYTHONPATH:./internal/ml/model_selection
conda activate trails




# ====================================
# ====================================
# determine the K and U tradeoff
# ====================================
# ====================================
# frappe
python internal/ml/model_selection/exps/micro/benchmark_ku.py \
  --search_space mlp_sp \
  --epoch 20 \
  --hidden_choice_len 20 \
  --dataset frappe \
  --base_dir ../exp_data/ \
  --only_phase1 True \
  --is_simulate True \
  --log_folder log_ku_tradeoff


# uci
python internal/ml/model_selection/exps/micro/benchmark_ku.py \
  --search_space mlp_sp \
  --hidden_choice_len 20 \
  --epoch 5 \
  --dataset uci_diabetes \
  --base_dir ../exp_data/ \
  --only_phase1 True \
  --is_simulate True \
  --log_folder log_ku_tradeoff

# criteo
python internal/ml/model_selection/exps/micro/benchmark_ku.py \
  --search_space mlp_sp \
  --hidden_choice_len 10 \
  --epoch 10 \
  --dataset criteo \
  --base_dir ../exp_data/ \
  --only_phase1 True \
  --is_simulate True \
  --log_folder log_ku_tradeoff


# c10
python internal/ml/model_selection/exps/micro/benchmark_ku.py \
  --search_space nasbench201 \
  --api_loc NAS-Bench-201-v1_1-096897.pth \
  --epoch 200 \
  --dataset cifar10 \
  --base_dir ../exp_data/ \
  --only_phase1 True \
  --is_simulate True \
  --log_folder log_ku_tradeoff


# c100
python internal/ml/model_selection/exps/micro/benchmark_ku.py \
  --search_space nasbench201 \
  --api_loc NAS-Bench-201-v1_1-096897.pth \
  --epoch 200 \
  --dataset cifar100 \
  --base_dir ../exp_data/ \
  --only_phase1 True \
  --is_simulate True \
  --log_folder log_ku_tradeoff


# imageNet
python internal/ml/model_selection/exps/micro/benchmark_ku.py \
  --search_space nasbench201 \
  --api_loc NAS-Bench-201-v1_1-096897.pth \
  --epoch 200 \
  --dataset ImageNet16-120 \
  --base_dir ../exp_data/ \
  --only_phase1 True \
  --is_simulate True \
  --log_folder log_ku_tradeoff



# ====================================
# ====================================
# determine the K and U tradeoff
# ====================================
# ====================================


python internal/ml/model_selection/exps/micro/benchmark_nk.py \
  --search_space mlp_sp \
  --epoch 20 \
  --hidden_choice_len 20 \
  --dataset frappe \
  --base_dir ../exp_data/ \
  --only_phase1 True \
  --is_simulate True \
  --log_folder log_ku_tradeoff


#uci
python internal/ml/model_selection/exps/micro/benchmark_nk.py \
  --search_space mlp_sp \
  --hidden_choice_len 20 \
  --epoch 5 \
  --dataset uci_diabetes \
  --base_dir ../exp_data/ \
  --only_phase1 True \
  --is_simulate True \
  --log_folder log_ku_tradeoff


# criteo
python internal/ml/model_selection/exps/micro/benchmark_nk.py \
  --search_space mlp_sp \
  --hidden_choice_len 10 \
  --epoch 10 \
  --dataset criteo \
  --base_dir ../exp_data/ \
  --only_phase1 True \
  --is_simulate True \
  --log_folder log_ku_tradeoff



# c10
python internal/ml/model_selection/exps/micro/benchmark_nk.py \
  --search_space nasbench201 \
  --api_loc NAS-Bench-201-v1_1-096897.pth \
  --epoch 200 \
  --dataset cifar10 \
  --base_dir ../exp_data/ \
  --only_phase1 True \
  --is_simulate True \
  --log_folder log_ku_tradeoff


# c100
python internal/ml/model_selection/exps/micro/benchmark_nk.py \
  --search_space nasbench201 \
  --api_loc NAS-Bench-201-v1_1-096897.pth \
  --epoch 200 \
  --dataset cifar100 \
  --base_dir ../exp_data/ \
  --only_phase1 True \
  --is_simulate True \
  --log_folder log_ku_tradeoff


# imageNet
python internal/ml/model_selection/exps/micro/benchmark_nk.py \
  --search_space nasbench201 \
  --api_loc NAS-Bench-201-v1_1-096897.pth \
  --epoch 200 \
  --dataset ImageNet16-120 \
  --base_dir ../exp_data/ \
  --only_phase1 True \
  --is_simulate True \
  --log_folder log_ku_tradeoff

