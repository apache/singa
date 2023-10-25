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



############## c10 dataset ##############
# run both 2phase-MS and training-free MS
python internal/ml/model_selection/exps/macro/anytime_img.py \
      --search_space nasbench201 \
      --api_loc NAS-Bench-201-v1_1-096897.pth \
      --epoch 200 \
      --dataset cifar10 \
      --num_labels 10 \
      --base_dir ../exp_data/ \
      --result_dir ./internal/ml/model_selection/exp_result/


############## c100 dataset ##############
python internal/ml/model_selection/exps/macro/anytime_img.py \
      --search_space nasbench201 \
      --api_loc NAS-Bench-201-v1_1-096897.pth \
      --epoch 200 \
      --dataset cifar100 \
      --num_labels 100 \
      --base_dir ../exp_data/ \
      --result_dir ./internal/ml/model_selection/exp_result/


############## imageNet dataset ##############
python internal/ml/model_selection/exps/macro/anytime_img.py \
      --search_space nasbench201 \
      --api_loc NAS-Bench-201-v1_1-096897.pth \
      --epoch 200 \
      --dataset ImageNet16-120 \
      --num_labels 120 \
      --base_dir ../exp_data/ \
      --result_dir ./internal/ml/model_selection/exp_result/



