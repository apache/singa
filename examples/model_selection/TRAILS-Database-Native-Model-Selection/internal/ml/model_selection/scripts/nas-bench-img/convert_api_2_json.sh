
export PYTHONPATH=$PYTHONPATH:./internal/ml/model_selection
conda activate trails


# pip install nats_bench

python internal/ml/model_selection/exps/nas_bench_img/0_characterize_gt.py
python internal/ml/model_selection/exps/nas_bench_img/0_parse_testacc_101.py
python internal/ml/model_selection/exps/nas_bench_img/0_parse_testacc_201.py


