

export PYTHONPATH=$PYTHONPATH:./internal/ml/model_selection

# rs
python internal/ml/model_selection/exps/baseline/train_with_random.py \
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


# rl
python internal/ml/model_selection/exps/baseline/train_with_rl.py


# re
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

# bohb
python internal/ml/model_selection/exps/baseline/train_bohb.py

############## draw the graph ##############
python internal/ml/model_selection/exps/baseline/draw_benchmark_train_based.py  --dataset frappe


