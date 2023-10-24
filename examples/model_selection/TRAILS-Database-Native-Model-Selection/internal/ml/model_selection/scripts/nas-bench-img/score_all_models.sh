


export PYTHONPATH=$PYTHONPATH:./internal/ml/model_selection


for i in {1..4}
do
    # cifar10 + nb101
#    /home/xingnaili/miniconda3/envs/trails/bin/python ./internal/ml/model_selection/exps/nas_bench_tabular/4.seq_score_online.py \
#      --models_explore=1200 \
#      --search_space=nasbench101 \
#      --api_loc=nasbench_only108.pkl \
#      --base_dir=/hdd1/xingnaili/exp_data/ \
#      --dataset=cifar10 \
#      --batch_size=32 \
#      --num_labels=10 \
#      --device=cuda:0 \
#      --log_folder=log_score_all_img10_101 \
#      --result_dir=./internal/ml/model_selection/exp_result/

    # cifar10 + nb201
    /home/xingnaili/miniconda3/envs/trails/bin/python ./internal/ml/model_selection/exps/nas_bench_tabular/4.seq_score_online.py \
      --models_explore=1200 \
      --search_space=nasbench201 \
      --api_loc=NAS-Bench-201-v1_1-096897.pth \
      --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset=cifar10 \
      --batch_size=32 \
      --num_labels=10 \
      --device=cpu \
      --log_folder=log_score_all_img10 \
      --result_dir=./internal/ml/model_selection/exp_result/

    # cifar100 + nb201
    /home/xingnaili/miniconda3/envs/trails/bin/python ./internal/ml/model_selection/exps/nas_bench_tabular/4.seq_score_online.py \
      --models_explore=1200 \
      --search_space=nasbench201 \
      --api_loc=NAS-Bench-201-v1_1-096897.pth \
      --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset=cifar100 \
      --batch_size=32 \
      --num_labels=100 \
      --device=cpu \
      --log_folder=log_score_all_img100 \
      --result_dir=./internal/ml/model_selection/exp_result/

    # imgnet + nb201
    /home/xingnaili/miniconda3/envs/trails/bin/python ./internal/ml/model_selection/exps/nas_bench_tabular/4.seq_score_online.py \
      --models_explore=1200 \
      --search_space=nasbench201 \
      --api_loc=NAS-Bench-201-v1_1-096897.pth \
      --base_dir=/hdd1/xingnaili/exp_data/ \
      --dataset=ImageNet16-120 \
      --batch_size=32 \
      --num_labels=120 \
      --device=cpu \
      --log_folder=log_score_all_img_imgnet \
      --result_dir=./internal/ml/model_selection/exp_result/
done