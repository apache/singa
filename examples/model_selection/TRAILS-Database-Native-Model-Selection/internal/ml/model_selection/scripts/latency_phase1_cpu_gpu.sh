

export PYTHONPATH=$PYTHONPATH:./internal/ml/model_selection


# frappe
python3 ./internal/ml/model_selection/exps/micro/benchmark_filtering_latency.py \
  --embedding_cache_filtering=False \
  --tfmem=express_flow \
  --models_explore=5000 \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=20 \
  --base_dir=/hdd1/xingnaili/exp_data/ \
  --num_labels=2 \
  --device=cuda:0 \
  --batch_size=32 \
  --dataset=frappe \
  --nfeat=5500 \
  --nfield=10 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result_sever_wo_cache/

#criteo
python3 ./internal/ml/model_selection/exps/micro/benchmark_filtering_latency.py \
  --embedding_cache_filtering=False \
  --tfmem=express_flow \
  --models_explore=5000 \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=10 \
  --base_dir=/hdd1/xingnaili/exp_data/ \
  --num_labels=2 \
  --device=cuda:0 \
  --batch_size=32 \
  --dataset=criteo \
  --nfeat=2100000 \
  --nfield=39 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result_sever_wo_cache/

# uci
python3 ./internal/ml/model_selection/exps/micro/benchmark_filtering_latency.py \
  --embedding_cache_filtering=False \
  --tfmem=express_flow \
  --models_explore=5000 \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=20 \
  --base_dir=/hdd1/xingnaili/exp_data/ \
  --num_labels=2 \
  --device=cuda:0 \
  --batch_size=32 \
  --dataset=uci_diabetes \
  --nfeat=369 \
  --nfield=43 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result_sever_wo_cache/


# cifar 10
python3 ./internal/ml/model_selection/exps/micro/benchmark_filtering_latency.py \
  --embedding_cache_filtering=False \
  --tfmem=express_flow \
  --models_explore=5000 \
  --search_space=nasbench201 \
  --api_loc=NAS-Bench-201-v1_1-096897.pth \
  --base_dir=/hdd1/xingnaili/exp_data/ \
  --num_labels=10 \
  --device=cuda:0 \
  --batch_size=32 \
  --dataset=cifar10 \
  --result_dir=./internal/ml/model_selection/exp_result_sever_wo_cache/


# cifar 100
python3 ./internal/ml/model_selection/exps/micro/benchmark_filtering_latency.py \
  --embedding_cache_filtering=False \
  --tfmem=express_flow \
  --models_explore=5000 \
  --search_space=nasbench201 \
  --api_loc=NAS-Bench-201-v1_1-096897.pth \
  --base_dir=/hdd1/xingnaili/exp_data/ \
  --num_labels=100 \
  --device=cuda:0 \
  --batch_size=32 \
  --dataset=cifar100 \
  --result_dir=./internal/ml/model_selection/exp_result_sever_wo_cache/


# imageNet
python3 ./internal/ml/model_selection/exps/micro/benchmark_filtering_latency.py \
  --embedding_cache_filtering=False \
  --tfmem=express_flow \
  --models_explore=5000 \
  --search_space=nasbench201 \
  --api_loc=NAS-Bench-201-v1_1-096897.pth \
  --base_dir=/hdd1/xingnaili/exp_data/ \
  --num_labels=120 \
  --device=cuda:0 \
  --batch_size=32 \
  --dataset=ImageNet16-120 \
  --result_dir=./internal/ml/model_selection/exp_result_sever_wo_cache/

########################## CPU ##############################
# this is run on cpu, only change the device==cpu for all above

# frappe
python3 ./internal/ml/model_selection/exps/micro/benchmark_filtering_latency.py \
  --embedding_cache_filtering=False \
  --tfmem=express_flow \
  --models_explore=5000 \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=20 \
  --base_dir=/hdd1/xingnaili/exp_data/ \
  --num_labels=2 \
  --device=cpu \
  --batch_size=32 \
  --dataset=frappe \
  --nfeat=5500 \
  --nfield=10 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result_sever_wo_cache/

# criteo
python3 ./internal/ml/model_selection/exps/micro/benchmark_filtering_latency.py \
  --embedding_cache_filtering=False \
  --tfmem=express_flow \
  --models_explore=5000 \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=10 \
  --base_dir=/hdd1/xingnaili/exp_data/ \
  --num_labels=2 \
  --device=cpu \
  --batch_size=32 \
  --dataset=criteo \
  --nfeat=2100000 \
  --nfield=39 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result_sever_wo_cache/

# uci
python3 ./internal/ml/model_selection/exps/micro/benchmark_filtering_latency.py \
  --embedding_cache_filtering=False \
  --tfmem=express_flow \
  --models_explore=5000 \
  --search_space=mlp_sp \
  --num_layers=4 \
  --hidden_choice_len=20 \
  --base_dir=/hdd1/xingnaili/exp_data/ \
  --num_labels=2 \
  --device=cpu \
  --batch_size=32 \
  --dataset=uci_diabetes \
  --nfeat=369 \
  --nfield=43 \
  --nemb=10 \
  --workers=0 \
  --result_dir=./internal/ml/model_selection/exp_result_sever_wo_cache/


# cifar 10
python3 ./internal/ml/model_selection/exps/micro/benchmark_filtering_latency.py \
  --embedding_cache_filtering=False \
  --tfmem=express_flow \
  --models_explore=5000 \
  --search_space=nasbench201 \
  --api_loc=NAS-Bench-201-v1_1-096897.pth \
  --base_dir=/hdd1/xingnaili/exp_data/ \
  --num_labels=10 \
  --device=cpu \
  --batch_size=32 \
  --dataset=cifar10 \
  --result_dir=./internal/ml/model_selection/exp_result_sever_wo_cache/


# cifar 100
python3 ./internal/ml/model_selection/exps/micro/benchmark_filtering_latency.py \
  --embedding_cache_filtering=False \
  --tfmem=express_flow \
  --models_explore=5000 \
  --search_space=nasbench201 \
  --api_loc=NAS-Bench-201-v1_1-096897.pth \
  --base_dir=/hdd1/xingnaili/exp_data/ \
  --num_labels=100 \
  --device=cpu \
  --batch_size=32 \
  --dataset=cifar100 \
  --result_dir=./internal/ml/model_selection/exp_result_sever_wo_cache/


# imageNet
python3 ./internal/ml/model_selection/exps/micro/benchmark_filtering_latency.py \
  --embedding_cache_filtering=False \
  --tfmem=express_flow \
  --models_explore=5000 \
  --search_space=nasbench201 \
  --api_loc=NAS-Bench-201-v1_1-096897.pth \
  --base_dir=/hdd1/xingnaili/exp_data/ \
  --num_labels=120 \
  --device=cpu \
  --batch_size=32 \
  --dataset=ImageNet16-120 \
  --result_dir=./internal/ml/model_selection/exp_result_sever_wo_cache/
