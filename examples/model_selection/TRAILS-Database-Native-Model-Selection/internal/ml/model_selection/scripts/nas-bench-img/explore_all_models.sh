
export PYTHONPATH=$PYTHONPATH:./internal/ml/model_selection
conda activate trails


# cifar10 + nb101
python ./internal/ml/model_selection/exps/nas_bench_img/1_explore_models_100_run.py \
  --search_space=nasbench101 \
  --api_loc=nasbench_only108.pkl \
  --base_dir=../exp_data/ \
  --dataset=cifar10 \
  --num_labels=10 \
  --device=cpu \
  --log_folder=log_img_explore_ea \
  --result_dir=./internal/ml/model_selection/exp_result/


# cifar10 + nb201
python ./internal/ml/model_selection/exps/nas_bench_img/1_explore_models_100_run.py \
  --search_space=nasbench201 \
  --api_loc=NAS-Bench-201-v1_1-096897.pth \
  --base_dir=../exp_data/ \
  --dataset=cifar10 \
  --init_channels=16 \
  --num_stacks=3 \
  --num_modules_per_stack=3 \
  --num_labels=10 \
  --device=cpu \
  --log_folder=log_img_explore_ea \
  --result_dir=./internal/ml/model_selection/exp_result/


# cifar100 + nb201
python ./internal/ml/model_selection/exps/nas_bench_img/1_explore_models_100_run.py \
  --search_space=nasbench201 \
  --api_loc=NAS-Bench-201-v1_1-096897.pth \
  --base_dir=../exp_data/ \
  --dataset=cifar100 \
  --init_channels=16 \
  --num_stacks=3 \
  --num_modules_per_stack=3 \
  --num_labels=100 \
  --device=cpu \
  --log_folder=log_img_explore_ea \
  --result_dir=./internal/ml/model_selection/exp_result/


# imgnet + nb201
python ./internal/ml/model_selection/exps/nas_bench_img/1_explore_models_100_run.py \
  --search_space=nasbench201 \
  --api_loc=NAS-Bench-201-v1_1-096897.pth \
  --base_dir=../exp_data/ \
  --dataset=ImageNet16-120 \
  --init_channels=16 \
  --num_stacks=3 \
  --num_modules_per_stack=3 \
  --num_labels=120 \
  --device=cpu \
  --log_folder=log_img_explore_ea \
  --result_dir=./internal/ml/model_selection/exp_result/
