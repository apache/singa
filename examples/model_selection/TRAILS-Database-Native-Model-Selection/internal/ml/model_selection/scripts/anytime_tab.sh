

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



