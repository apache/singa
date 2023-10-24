

export PYTHONPATH=$PYTHONPATH:./internal/ml/model_selection
conda activate trails


nohup python ./internal/ml/model_selection/exps/nas_bench_tabular/0.train_one_model.py  \
  --log_name=baseline_train_based \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=20 \
  --base_dir=../exp_data/ \
  --num_labels=2 \
  --device=cuda:0 \
  --batch_size=1024 \
  --lr=0.001 \
  --epoch=3 \
  --iter_per_epoch=200 \
  --dataset=uci_diabetes \
  --nfeat=369 \
  --nfield=43 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result/ \
  --log_folder=log_uci_train_tune >uci_3.log &



nohup  python ./internal/ml/model_selection/exps/nas_bench_tabular/0.train_one_model.py  \
  --log_name=baseline_train_based \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=20 \
  --base_dir=../exp_data/ \
  --num_labels=2 \
  --device=cuda:1 \
  --batch_size=1024 \
  --lr=0.001 \
  --epoch=5 \
  --iter_per_epoch=200 \
  --dataset=uci_diabetes \
  --nfeat=369 \
  --nfield=43 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result/ \
  --log_folder=log_uci_train_tune >uci_5.log &


# default setting.
nohup  python ./internal/ml/model_selection/exps/nas_bench_tabular/0.train_one_model.py  \
  --log_name=baseline_train_based \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=20 \
  --base_dir=../exp_data/ \
  --num_labels=2 \
  --device=cuda:2 \
  --batch_size=1024 \
  --lr=0.001 \
  --epoch=7 \
  --iter_per_epoch=200 \
  --dataset=uci_diabetes \
  --nfeat=369 \
  --nfield=43 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result/ \
  --log_folder=log_uci_train_tune >uci_7.log &

