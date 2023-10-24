

export PYTHONPATH=$PYTHONPATH:./internal/ml/model_selection
conda activate trails



############## c10 dataset ##############
# run both 2phase-MS and training-free MS
python internal/ml/model_selection/exps/micro/benchmark_budget_aware_alg.py \
      --search_space nasbench201 \
      --api_loc NAS-Bench-201-v1_1-096897.pth \
      --dataset cifar10 \
      --epoch 200 \
      --base_dir ../exp_data/ \
      --log_name logs_default \
      --result_dir ./internal/ml/model_selection/exp_result/


############## c100 dataset ##############
python internal/ml/model_selection/exps/micro/benchmark_budget_aware_alg.py \
      --search_space nasbench201 \
      --api_loc NAS-Bench-201-v1_1-096897.pth \
      --dataset cifar100 \
      --epoch 200 \
      --base_dir ../exp_data/ \
      --log_name logs_default \
      --result_dir ./internal/ml/model_selection/exp_result/


############## imageNet dataset ##############
python internal/ml/model_selection/exps/micro/benchmark_budget_aware_alg.py \
      --search_space nasbench201 \
      --api_loc NAS-Bench-201-v1_1-096897.pth \
      --dataset ImageNet16-120 \
      --epoch 200 \
      --base_dir ../exp_data/ \
      --log_name logs_default \
      --result_dir ./internal/ml/model_selection/exp_result/



############## draw graphs ##############
python internal/ml/model_selection/exps/micro/draw_budget_aware_alg.py
