

export PYTHONPATH=$PYTHONPATH:./internal/ml/model_selection
conda activate trails



# run both training-based MS
############## c10 dataset ##############
python internal/ml/model_selection/exps/baseline/train_with_ea.py \
  --search_space nasbench201 \
  --api_loc NAS-Bench-201-v1_1-096897.pth \
  --epoch 200 \
  --dataset cifar10 \
  --num_labels 10 \
  --base_dir ../exp_data/ \
  --log_folder log_baseline_c10 \
  --result_dir ./internal/ml/model_selection/exp_result/


############## c100 dataset ##############
python internal/ml/model_selection/exps/baseline/train_with_ea.py \
  --search_space nasbench201 \
  --api_loc NAS-Bench-201-v1_1-096897.pth \
  --epoch 200 \
  --dataset cifar100 \
  --num_labels 100 \
  --base_dir ../exp_data/ \
  --log_folder log_baseline_c100 \
  --result_dir ./internal/ml/model_selection/exp_result/


############## ImgNet dataset ##############
python internal/ml/model_selection/exps/baseline/train_with_ea.py \
  --search_space nasbench201 \
  --api_loc NAS-Bench-201-v1_1-096897.pth \
  --epoch 200 \
  --dataset ImageNet16-120 \
  --num_labels 120 \
  --base_dir ../exp_data/ \
  --log_folder log_baseline_imgnet \
  --result_dir ./internal/ml/model_selection/exp_result/



