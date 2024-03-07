# Test Singa for model selection

Run those three functions to ensure the singa can run.

```bash
python3 ./internal/ml/model_selection/exps/4.seq_score_online.py   --embedding_cache_filtering=True   --models_explore=10   --tfmem=synflow   --log_name=score_based   --search_space=mlp_sp   --num_layers=4   --hidden_choice_len=20   --base_dir=./dataset   --num_labels=2   --device=cpu   --batch_size=32   --dataset=frappe   --nfeat=5500   --nfield=10   --nemb=10   --workers=0   --result_dir=./exp_result/   --log_folder=log_foler

python3 ./internal/ml/model_selection/exps/0.train_one_model.py --log_name=train_log --search_space=mlp_sp --base_dir=./dataset --num_labels=2 --device=cpu --batch_size=10 --lr=0.01 --epoch=5 --iter_per_epoch=2000 --dataset=frappe --nfeat=5500 --nfield=10 --nemb=10 --workers=0 --result_dir=./exp_result/   --log_folder=log_foler

python3 internal/ml/model_selection/pg_interface.py
```

