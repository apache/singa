

export PYTHONPATH=$PYTHONPATH:./internal/ml/model_selection
conda activate trails


worker_id=0
GPU_NUM=8
worker_each_gpu=16
total_workers=$((worker_each_gpu*GPU_NUM))

for((gpu_id=0; gpu_id < GPU_NUM; ++gpu_id)); do
#  echo "GPU id is $gpu_id"
  for((i=0; i < worker_each_gpu; ++i)); do
    echo "nohup python ./internal/ml/model_selection/exps/nas_bench_tabular/2.seq_train_online.py  \
    --log_name=baseline_train_based \
    --search_space=mlp_sp \
    --num_layers=4 \
    --hidden_choice_len=20 \
    --base_dir=/home/shaofeng/naili/firmest_data/ \
    --num_labels=2 \
    --device=cuda:$gpu_id \
    --batch_size=512 \
    --lr=0.001 \
    --epoch=20 \
    --iter_per_epoch=200 \
    --dataset=frappe \
    --nfeat=5500 \
    --nfield=10 \
    --nemb=10 \
    --worker_id=$worker_id \
    --total_workers=$total_workers  \
    --workers=0 \
    --log_folder=log_frappe \
    --total_models_per_worker=-1 \
    --result_dir=./internal/ml/model_selection/exp_result/ \
    --pre_partitioned_file=./internal/ml/model_selection/exps/nas_bench_tabular/sampled_models_all.json & ">> train_all_models_frappe_seq.sh

    sleep 1
    worker_id=$((worker_id+1))
  done
done


# pkill -9 -f internal/ml/model_selection/scripts/nas-bench-tabular/train_all_models_frappe.sh
