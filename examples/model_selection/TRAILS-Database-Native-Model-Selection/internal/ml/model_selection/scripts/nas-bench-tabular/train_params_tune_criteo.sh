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



# default setting.
python ./internal/ml/model_selection/exps/nas_bench_tabular/0.train_one_model.py  \
  --log_name=baseline_train_based \
  --search_space=mlp_sp \
  --base_dir=../exp_data/ \
  --num_labels=2 \
  --device=cuda:0 \
  --batch_size=1024 \
  --lr=0.001 \
  --epoch=5 \
  --iter_per_epoch=2000 \
  --dataset=criteo \
  --nfeat=2100000 \
  --nfield=39 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result/ \
  --log_folder=log_criteo_train_tune >criteo_5.log &


python ./internal/ml/model_selection/exps/nas_bench_tabular/0.train_one_model.py  \
  --log_name=baseline_train_based \
  --search_space=mlp_sp \
  --base_dir=../exp_data/ \
  --num_labels=2 \
  --device=cuda:0 \
  --batch_size=1024 \
  --lr=0.001 \
  --epoch=10 \
  --iter_per_epoch=2000 \
  --dataset=criteo \
  --nfeat=2100000 \
  --nfield=39 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result/ \
  --log_folder=log_criteo_train_tune >criteo_10.log &



python ./internal/ml/model_selection/exps/nas_bench_tabular/0.train_one_model.py  \
  --log_name=baseline_train_based \
  --search_space=mlp_sp \
  --base_dir=../exp_data/ \
  --num_labels=2 \
  --device=cuda:1 \
  --batch_size=1024 \
  --lr=0.001 \
  --epoch=20 \
  --iter_per_epoch=2000 \
  --dataset=criteo \
  --nfeat=2100000 \
  --nfield=39 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result/ \
  --log_folder=log_criteo_train_tune >criteo_20.log &




python ./internal/ml/model_selection/exps/nas_bench_tabular/0.train_one_model.py  \
  --log_name=baseline_train_based \
  --search_space=mlp_sp \
  --base_dir=../exp_data/ \
  --num_labels=2 \
  --device=cuda:2 \
  --batch_size=1024 \
  --lr=0.001 \
  --epoch=40 \
  --iter_per_epoch=2000 \
  --dataset=criteo \
  --nfeat=2100000 \
  --nfield=39 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result/ \
  --log_folder=log_criteo_train_tune >criteo_40.log &



python ./internal/ml/model_selection/exps/nas_bench_tabular/0.train_one_model.py  \
  --log_name=baseline_train_based \
  --search_space=mlp_sp \
  --base_dir=../exp_data/ \
  --num_labels=2 \
  --device=cuda:3 \
  --batch_size=1024 \
  --lr=0.001 \
  --epoch=60 \
  --iter_per_epoch=2000 \
  --dataset=criteo \
  --nfeat=2100000 \
  --nfield=39 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result/ \
  --log_folder=log_criteo_train_tune >criteo_60.log &



python ./internal/ml/model_selection/exps/nas_bench_tabular/0.train_one_model.py  \
  --log_name=baseline_train_based \
  --search_space=mlp_sp \
  --base_dir=../exp_data/ \
  --num_labels=2 \
  --device=cuda:4 \
  --batch_size=1024 \
  --lr=0.001 \
  --epoch=80 \
  --iter_per_epoch=2000 \
  --dataset=criteo \
  --nfeat=2100000 \
  --nfield=39 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result/ \
  --log_folder=log_criteo_train_tune >criteo_80.log &



python ./internal/ml/model_selection/exps/nas_bench_tabular/0.train_one_model.py  \
  --log_name=baseline_train_based \
  --search_space=mlp_sp \
  --base_dir=../exp_data/ \
  --num_labels=2 \
  --device=cuda:5 \
  --batch_size=1024 \
  --lr=0.001 \
  --epoch=100 \
  --iter_per_epoch=2000 \
  --dataset=criteo \
  --nfeat=2100000 \
  --nfield=39 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result/ \
  --log_folder=log_criteo_train_tune >criteo_100.log &

