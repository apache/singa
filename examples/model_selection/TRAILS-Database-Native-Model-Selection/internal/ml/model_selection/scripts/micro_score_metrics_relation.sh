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


############## Frappe ##############
# run both 2phase-MS and training-free MS
python ./internal/ml/model_selection/exps/micro/benchmark_score_metrics.py \
      --tfmem=express_flow \
      --search_space mlp_sp \
      --dataset frappe \
      --base_dir ../exp_data/ \
      --log_name logs_default \
      --result_dir ./internal/ml/model_selection/exp_result/


############## criteo dataset ##############
python ./internal/ml/model_selection/exps/micro/benchmark_score_metrics.py \
      --tfmem=express_flow \
      --search_space mlp_sp \
      --dataset criteo \
      --base_dir ../exp_data/ \
      --log_name logs_default \
      --result_dir ./internal/ml/model_selection/exp_result/


############## Uci dataset ##############
python ./internal/ml/model_selection/exps/micro/benchmark_score_metrics.py \
      --tfmem=express_flow \
      --search_space=mlp_sp \
      --dataset uci_diabetes \
      --base_dir ../exp_data/ \
      --log_name logs_default \
      --result_dir ./internal/ml/model_selection/exp_result/


############## draw graphs ##############
python ./internal/ml/model_selection/exps/micro/draw_score_metric_relation.py
