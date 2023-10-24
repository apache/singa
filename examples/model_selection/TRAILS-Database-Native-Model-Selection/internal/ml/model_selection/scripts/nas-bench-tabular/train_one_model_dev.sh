

export PYTHONPATH=$PYTHONPATH:./internal/ml/model_selection

python ./internal/ml/model_selection/exps/nas_bench_tabular/0.train_one_model.py  \
    --log_name=baseline_train_based \
    --search_space=mlp_sp \
    --num_layers=4 \
    --hidden_choice_len=20 \
    --base_dir=/hdd1/xingnaili/exp_data/ \
    --num_labels=2 \
    --device=cuda:0 \
    --batch_size=512 \
    --lr=0.001 \
    --epoch=20 \
    --iter_per_epoch=200 \
    --dataset=frappe \
    --nfeat=5500 \
    --nfield=10 \
    --nemb=10 \
    --worker_id=0 \
    --total_workers=1 \
    --workers=1 \
    --result_dir=./internal/ml/model_selection/exp_result/ \
    --log_folder=log_frappe