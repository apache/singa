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


############## frappe dataset ##############

# run the 2phase-MS
python internal/ml/model_selection/exps/macro/anytime_simulate.py \
      --search_space mlp_sp \
      --num_layers 4 \
      --hidden_choice_len 20 \
      --batch_size 128 \
      --nfeat 5500 \
      --nfield 10 \
        --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset frappe \
      --num_labels 2 \
      --only_phase1 False \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_frappe \
      --result_dir ./internal/ml/model_selection/exp_result/ \
      --num_points 5


# run the training-free MS
python internal/ml/model_selection/exps/macro/anytime_simulate.py \
      --search_space mlp_sp \
      --num_layers 4 \
      --hidden_choice_len 20 \
      --batch_size 128 \
      --nfeat 5500 \
      --nfield 10 \
        --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset frappe \
      --num_labels 2 \
      --only_phase1 True \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_frappe \
      --result_dir ./internal/ml/model_selection/exp_result/ \
      --num_points 5


############## uci dataset ##############

# run the 2phase-MS
python internal/ml/model_selection/exps/macro/anytime_simulate.py \
      --search_space mlp_sp \
      --num_layers 4 \
      --hidden_choice_len 20 \
      --batch_size 128 \
      --nfeat 369 \
      --nfield 43 \
        --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset uci_diabetes \
      --num_labels 2 \
      --only_phase1 False \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_uci_diabetes \
      --result_dir ./internal/ml/model_selection/exp_result/ \
      --num_points 5


# run the training-free MS
python internal/ml/model_selection/exps/macro/anytime_simulate.py \
      --search_space mlp_sp \
      --num_layers 4 \
      --hidden_choice_len 20 \
      --batch_size 128 \
      --nfeat 369 \
      --nfield 43 \
        --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset uci_diabetes \
      --num_labels 2 \
      --only_phase1 True \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_uci_diabetes \
      --result_dir ./internal/ml/model_selection/exp_result/ \
      --num_points 5


############## criteo dataset ##############

# run the 2phase-MS
python internal/ml/model_selection/exps/macro/anytime_simulate.py \
      --search_space mlp_sp \
      --num_layers 4 \
      --hidden_choice_len 10 \
      --batch_size 128 \
      --nfeat 2100000 \
      --nfield 39 \
        --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset criteo \
      --num_labels 2 \
      --only_phase1 False \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_criteo \
      --result_dir ./internal/ml/model_selection/exp_result/ \
      --num_points 5


# run the training-free MS
python internal/ml/model_selection/exps/macro/anytime_simulate.py \
      --search_space mlp_sp \
      --num_layers 4 \
      --hidden_choice_len 10 \
      --batch_size 128 \
      --nfeat 2100000 \
      --nfield 39 \
        --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset criteo \
      --num_labels 2 \
      --only_phase1 True \
      --is_simulate True \
      --device cpu \
      --log_folder any_time_criteo \
      --result_dir ./internal/ml/model_selection/exp_result/ \
      --num_points 5



