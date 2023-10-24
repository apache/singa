

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
