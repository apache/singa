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



# run both training-based MS
############## frappe dataset ##############
python internal/ml/model_selection/exps/baseline/train_with_ea.py \
  --search_space mlp_sp \
  --num_layers 4 \
  --hidden_choice_len 20 \
  --epoch 19 \
  --batch_size=512 \
  --lr=0.001 \
  --iter_per_epoch=200 \
  --nfeat=5500 \
  --nfield=10 \
  --nemb=10 \
  --base_dir ../exp_data/ \
  --dataset frappe \
  --num_labels 2 \
  --device=cpu \
  --log_folder baseline_frappe \
  --result_dir ./internal/ml/model_selection/exp_result/


############## uci dataset ##############
python internal/ml/model_selection/exps/baseline/train_with_ea.py \
  --search_space mlp_sp \
  --num_layers 4 \
  --hidden_choice_len 20 \
  --epoch 0 \
  --batch_size=1024 \
  --lr=0.001 \
  --iter_per_epoch=200 \
  --nfeat=369 \
  --nfield=43 \
  --nemb=10 \
  --base_dir ../exp_data/ \
  --dataset uci_diabetes \
  --num_labels 2 \
  --device=cpu \
  --log_folder baseline_uci_diabetes \
  --result_dir ./internal/ml/model_selection/exp_result/


############## criteo dataset ##############
python internal/ml/model_selection/exps/baseline/train_with_ea.py \
  --search_space mlp_sp \
  --num_layers 4 \
  --hidden_choice_len 10 \
  --epoch 9 \
  --batch_size=1024 \
  --lr=0.001 \
  --iter_per_epoch=2000 \
  --nfeat=2100000 \
  --nfield=39 \
  --nemb=10 \
  --base_dir ../exp_data/ \
  --dataset criteo \
  --num_labels 2 \
  --device=cpu \
  --log_folder baseline_criteo \
  --result_dir ./internal/ml/model_selection/exp_result/

