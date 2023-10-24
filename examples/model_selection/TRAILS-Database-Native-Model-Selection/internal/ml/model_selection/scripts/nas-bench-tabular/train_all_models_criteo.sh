
export PYTHONPATH=$PYTHONPATH:./internal/ml/model_selection
conda activate trails


worker_id=0
GPU_NUM=9
worker_each_gpu=6
total_workers=$((worker_each_gpu*GPU_NUM))

for((gpu_id=0; gpu_id < GPU_NUM; ++gpu_id)); do
#  echo "GPU id is $gpu_id"
  for((i=0; i < worker_each_gpu; ++i)); do
    echo "Assign task to worker id is $worker_id"
    echo "nohup python ./internal/ml/model_selection/exps/nas_bench_tabular/2.seq_train_online.py  \
    --log_name=baseline_train_based \
    --search_space=mlp_sp \
    --num_layers=4 \
    --hidden_choice_len=10 \
    --base_dir=../exp_data/ \
    --num_labels=2 \
    --device=cuda:$gpu_id \
    --batch_size=1024 \
    --lr=0.001 \
    --epoch=10 \
    --iter_per_epoch=2000 \
    --dataset=criteo \
    --nfeat=2100000 \
    --nfield=39 \
    --nemb=10 \
    --worker_id=$worker_id \
    --total_workers=$total_workers \
    --workers=0 \
    --log_folder=log_train_criteo \
    --total_models_per_worker=-1 \
    --result_dir=./internal/ml/model_selection/exp_result/ \
    --pre_partitioned_file=./internal/ml/model_selection/exps/nas_bench_tabular/sampled_models_10000_models.json & ">> train_all_models_criteo_seq.sh

#    sleep 1
    worker_id=$((worker_id+1))
  done
done


# pkill -9 -f 2.seq_train_online.py
# run with bash internal/ml/model_selection/scripts/nas-bench-tabular/train_all_models_criteo.sh >criteobash &
