

export PYTHONPATH=$PYTHONPATH:./internal/ml/model_selection
conda activate trails




python ./internal/ml/model_selection/exps/nas_bench_tabular/4.seq_score_online.py \
  --models_explore=1000 \
  --log_name=score_based \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=10 \
  --base_dir=/hdd1/xingnaili/exp_data/ \
  --num_labels=2 \
  --device=cuda:6 \
  --batch_size=32 \
  --dataset=criteo \
  --nfeat=2100000 \
  --nfield=39 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result/ \
  --log_folder=log_score_time_criteo  > outputCriScorAll.log&






















