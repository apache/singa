

# frappe
python exps/main_v2/ground_truth/2.seq_train_dist_online.py  \
    --log_name=baseline_train_based \
    --search_space=mlp_sp \
    --num_layers=4 \
    --hidden_choice_len=20 \
    --base_dir=../exp_data/ \
    --num_labels=1 \
    --device=gpu \
    --batch_size=1024 \
    --lr=0.001 \
    --epoch=10 \
    --iter_per_epoch=100 \
    --dataset=frappe \
    --nfeat=5500 \
    --nfield=10 \
    --nemb=10 \
    --total_models_per_worker=10 \
    --workers=0 \
    --worker_each_gpu=1 \
    --gpu_num=8 \
    --log_folder=LogFrappee \
    --pre_partitioned_file=./exps/main_v2/ground_truth/sampled_models_10000_models.json &

# criteo
python exps/main_v2/ground_truth/2.seq_train_dist_online.py  \
    --log_name=baseline_train_based \
    --search_space=mlp_sp \
    --num_layers=4 \
    --hidden_choice_len=10 \
    --base_dir=../exp_data/ \
    --num_labels=1 \
    --device=gpu \
    --batch_size=1024 \
    --lr=0.001 \
    --epoch=10 \
    --iter_per_epoch=2000 \
    --dataset=criteo \
    --nfeat=2100000 \
    --nfield=39 \
    --nemb=10 \
    --workers=0 \
    --worker_each_gpu=9 \
    --gpu_num=8 \
    --log_folder=LogCriteo \
    --pre_partitioned_file=./exps/main_v2/ground_truth/sampled_models_10000_models.json &
