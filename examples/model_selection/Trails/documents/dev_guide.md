<!--
    Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with < this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.
-->

# Test Singa for model selection

Run those three functions to ensure the singa can run.

```bash
python3 ./internal/ml/model_selection/exps/4.seq_score_online.py   --embedding_cache_filtering=True   --models_explore=10   --tfmem=synflow   --log_name=score_based   --search_space=mlp_sp   --num_layers=4   --hidden_choice_len=20   --base_dir=./dataset   --num_labels=2   --device=cpu   --batch_size=32   --dataset=frappe   --nfeat=5500   --nfield=10   --nemb=10   --workers=0   --result_dir=./exp_result/   --log_folder=log_foler

python3 ./internal/ml/model_selection/exps/0.train_one_model.py --log_name=train_log --search_space=mlp_sp --base_dir=./dataset --num_labels=2 --device=cpu --batch_size=10 --lr=0.01 --epoch=5 --iter_per_epoch=2000 --dataset=frappe --nfeat=5500 --nfield=10 --nemb=10 --workers=0 --result_dir=./exp_result/   --log_folder=log_foler

python3 internal/ml/model_selection/pg_interface.py
```

# How to add features to there?

1. fork the git repo https://github.com/apache/singa/tree/dev-postgresql

2. run the docker image as in [README.md](https://github.com/apache/singa/blob/dev-postgresql/examples/model_selection/Trails/README.md)

   ```bash
   # Remove existing one if there is
   docker rm -f singa_trails
   # Create project folder.
   mkdir project && cd project
   # Download the Dockerile.
   wget -O Dockerfile https://raw.githubusercontent.com/apache/singa/dev-postgresql/examples/model_selection/Trails/singa.psql.Dockerfile

   # Build Dockerile and run the docker.
   docker build -t singa_trails .
   docker run -d --name singa_trails singa_trails
   # Wait for 5 mins, monitor the logs until it shows "Done!", then exit the monitor
   docker logs -f singa_trails
   ```

3. after the docker image is running up, go to the code dir, add your own git repo's url as a remote

   ```bash
   docker exec -it singa_trails bash
   cd /project/Trails
   git fetch --all
   git checkout -b <branch-name> <my-repo>/<branch-name>
   # eg git checkout -b dev-postgresql nl2/dev-postgresql
   ```

4. then commit the change to your own git repo, and pull inside the docker image.

5. then, compile and test as following

## For the psql

```bash
cd Trails/internal/pg_extension
cargo pgrx run --release

# tests
psql -h localhost -p 28814 -U postgres
\c pg_extension

# Test coordinator
SELECT coordinator('0.08244', '168.830156', '800', false, '/project/Trails/internal/ml/model_selection/config.ini');
# Run an example, wait one min, it will run filtering + refinemnt + training the selected model.
CALL model_selection_end2end('frappe_train', ARRAY['col1', 'col2', 'col3', 'col4','col5','col6','col7','col8','col9','col10', 'label'], '10', '/project/Trails/internal/ml/model_selection/config.ini');

# In other terminal, monitor the running process
docker exec -it trails_polardb bash
tail -f /home/postgres/.pgrx/data-14/trails_log_folder/<log_file_name>
```

## For the polarDB

```bash
cd Trails/internal/pg_extension
cargo clean
cargo pgrx install --pg-config /home/postgres/tmp_basedir_polardb_pg_1100_bld/bin/pg_config

# Connect to the primary pg server and use pg_extension database.
docker exec -it singa_trails_polardb bash
psql -h localhost -p 5432 -U postgres
\c pg_extension

# Test coordinator
SELECT coordinator('0.08244', '168.830156', '800', false, '/home/postgres/Trails/internal/ml/model_selection/config.ini');
# Run an example, wait one min, it will run filtering + refinemnt + training the selected model.
CALL model_selection_end2end('frappe_train', ARRAY['col1', 'col2', 'col3', 'col4','col5','col6','col7','col8','col9','col10', 'label'], '10', '/home/postgres/Trails/internal/ml/model_selection/config.ini');

# In other terminal, monitor the running process
docker exec -it singa_trails_polardb bash
tail -f /var/polardb/primary_datadir/trails_log_folder/<log_file_name>
```
