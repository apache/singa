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


worker_id=0
GPU_NUM=8
worker_each_gpu=4
total_workers=$((worker_each_gpu*GPU_NUM))

for((gpu_id=0; gpu_id < GPU_NUM; ++gpu_id)); do
  for((i=0; i < worker_each_gpu; ++i)); do

    echo "nohup python ./internal/ml/model_selection/exps/nas_bench_tabular/2.seq_train_online.py  \
    --log_name=baseline_train_based \
    --search_space=mlp_sp \
    --num_layers=4 \
    --hidden_choice_len=20 \
    --base_dir=../exp_data/ \
    --num_labels=2 \
    --device=cuda:$gpu_id \
    --batch_size=1024 \
    --lr=0.001 \
    --epoch=40 \
    --iter_per_epoch=200 \
    --dataset=uci_diabetes \
    --nfeat=369 \
    --nfield=43 \
    --nemb=10 \
    --worker_id=$worker_id \
    --total_workers=$total_workers \
    --workers=0 \
    --log_folder=log_train_uci  \
    --total_models_per_worker=-1 \
    --result_dir=./internal/ml/model_selection/exp_result/ \
    --pre_partitioned_file=./internal/ml/model_selection/exps/nas_bench_tabular/uci_left_8k_models.json > outputuci.log& ">> train_all_models_diabetes_seq.sh

    worker_id=$((worker_id+1))
  done
done


# pkill -9 -f ./internal/ml/model_selection/exps/nas_bench_tabular//2.seq_train_online.py
# pkill -9 -f /home/naili/miniconda3/envs/firmest_torch11/bin/python

# run with bash internal/ml/model_selection/scripts/nas-bench-tabular/train_all_models_diabetes.sh >ucibash &
