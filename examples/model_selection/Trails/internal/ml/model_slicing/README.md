#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


# Powering In-Database Dynamic Model Slicing for Structured Data Analytics

The general model based on LEADS is at [algorithm](https://github.com/Zrealshadow/SAMS/tree/f0570730563e7e05e073d5b7eaedabebe6286f56).

# Envs

```bash
pip install orjson
pip install einops
pip install tqdm
pip install matplotlib

unset PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/project/Trails/internal/ml/
export PYTHONPATH=$PYTHONPATH:/project/Trails/internal/ml/model_slicing/
export PYTHONPATH=$PYTHONPATH:/project/Trails/internal/ml/model_slicing/algorithm/
echo $PYTHONPATH


export PYTHONPATH=$PYTHONPATH:/home/xingnaili/Trails/internal/ml/
export PYTHONPATH=$PYTHONPATH:/home/xingnaili/Trails/internal/ml/model_slicing/
export PYTHONPATH=$PYTHONPATH:/home/xingnaili/Trails/internal/ml/model_slicing/algorithm/


/project/Trails/internal/ml/
/project/Trails/internal/ml/model_slicing/algorithm:
/project/Trails/internal/ml/model_slicing:

```

# Save data 

4 datasets are used here.

```
adult  bank  cvd  frappe  payment(credit)  credit(hcdr)  census  diabetes
```

Save the statistics

```bash
# save the data cardinalities, run in docker

# frappe
python3 ./internal/ml/model_slicing/algorithm/save_satistics.py --dataset frappe --data_dir /hdd1/sams/data/ --nfeat 5500 --nfield 10 --max_filter_col 10 --train_dir ./

# adult
python3 ./internal/ml/model_slicing/algorithm/save_satistics.py --dataset adult --data_dir /hdd1/sams/data/ --nfeat 140 --nfield 13 --max_filter_col 13 --train_dir ./

# cvd
python3 ./internal/ml/model_slicing/algorithm/save_satistics.py --dataset cvd --data_dir /hdd1/sams/data/ --nfeat 110 --nfield 11 --max_filter_col 11 --train_dir ./

# bank
python3 ./internal/ml/model_slicing/algorithm/save_satistics.py --dataset bank --data_dir /hdd1/sams/data/ --nfeat 80 --nfield 16 --max_filter_col 16 --train_dir ./



New Datasets
# census 
python3 ./internal/ml/model_slicing/algorithm/save_satistics.py --dataset census --data_dir /hdd1/sams/data/ --nfeat 540 --nfield 41 --max_filter_col 41  --train_dir ./

# Payment (credit)
python3 ./internal/ml/model_slicing/algorithm/save_satistics.py --dataset credit --data_dir /hdd1/sams/data/ --nfeat 350 --nfield 23 --max_filter_col 23 --train_dir ./

# diabetes 
python3 ./internal/ml/model_slicing/algorithm/save_satistics.py --dataset diabetes --data_dir /hdd1/sams/data/ --nfeat 850 --nfield 48 --max_filter_col 48 --train_dir ./

# credit (hcdr)
python3 ./internal/ml/model_slicing/algorithm/save_satistics.py --dataset hcdr --data_dir /hdd1/sams/data/ --nfeat 550 --nfield 69 --max_filter_col 69 --train_dir ./

```

# Run docker

```bash
# in server
ssh panda17

# goes to /home/xingnaili/firmest_docker/Trails
git submodule update --recursive --remote

# run container
docker run -d --name moe_inf \
  --network="host" \
  -v $(pwd)/Trails:/project/Trails \
  -v /hdd1/sams/tensor_log/:/project/tensor_log \
  -v /hdd1/sams/data/:/project/data_all \
  trails
    
# Enter the docker container.
docker exec -it moe_inf bash 
```



# 12 Run in database

Config the database runtime

```sql
cargo pgrx run --release
```

Load data into RDBMS

```bash

psql -h localhost -p 28814 -U postgres 
\l
\c pg_extension
\dt
\d frappe_train


# frappe
bash /project/Trails/internal/ml/model_selection/scripts/database/load_data_to_db.sh /project/data_all/frappe frappe
# frappe, only feature ids
bash /project/Trails/internal/ml/model_selection/scripts/database/load_data_to_db_int.sh /project/data_all/frappe frappe


# adult
bash ./internal/ml/model_selection/scripts/database/load_data_to_db.sh /project/data_all/adult adult
# adult, only feature ids
bash ./internal/ml/model_selection/scripts/database/load_data_to_db_int.sh /project/data_all/adult adult
# check type is correct or not. 
SELECT column_name, data_type, column_default, is_nullable 
FROM information_schema.columns 
WHERE table_name = 'adult_int_train';


# cvd 
bash /project/Trails/internal/ml/model_selection/scripts/database/load_data_to_db.sh /project/data_all/cvd cvd
# cvd, only feature ids
bash /project/Trails/internal/ml/model_selection/scripts/database/load_data_to_db_int.sh /project/data_all/cvd cvd


# bank
bash /project/Trails/internal/ml/model_selection/scripts/database/load_data_to_db.sh /project/data_all/bank bank
# bank, only feature ids
bash /project/Trails/internal/ml/model_selection/scripts/database/load_data_to_db_int.sh /project/data_all/bank bank


New Datasets

# census
bash /project/Trails/internal/ml/model_selection/scripts/database/load_data_to_db_int.sh /project/data_all/census census

# credit
bash /project/Trails/internal/ml/model_selection/scripts/database/load_data_to_db_int.sh /project/data_all/credit credit

# hcdr
bash /project/Trails/internal/ml/model_selection/scripts/database/load_data_to_db_int.sh /project/data_all/hcdr hcdr

# diabetes
bash /project/Trails/internal/ml/model_selection/scripts/database/load_data_to_db_int.sh /project/data_all/diabetes diabetes
```

Verify data is in the DB

```sql
# check table status
\dt
\d frappe_train
SELECT * FROM frappe_train LIMIT 10;
```

Config

```sql
# after run the pgrx, then edie the sql
# generate schema
cargo pgrx schema >> /home/postgres/.pgrx/14.9/pgrx-install/share/extension/pg_extension--0.1.0.sql


-- src/lib.rs:266
-- pg_extension::model_init
CREATE  FUNCTION "model_init"(
	"condition" TEXT, /* alloc::string::String */
	"config_file" TEXT, /* alloc::string::String */
	"col_cardinalities_file" TEXT, /* alloc::string::String */
	"model_path" TEXT /* alloc::string::String */
) RETURNS TEXT /* alloc::string::String */
IMMUTABLE STRICT PARALLEL SAFE 
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'model_init_wrapper';

-- src/lib.rs:242
-- pg_extension::inference_shared_write_once_int
CREATE  FUNCTION "inference_shared_write_once_int"(
	"dataset" TEXT, /* alloc::string::String */
	"condition" TEXT, /* alloc::string::String */
	"config_file" TEXT, /* alloc::string::String */
	"col_cardinalities_file" TEXT, /* alloc::string::String */
	"model_path" TEXT, /* alloc::string::String */
	"sql" TEXT, /* alloc::string::String */
	"batch_size" INT /* i32 */
) RETURNS TEXT /* alloc::string::String */
IMMUTABLE STRICT PARALLEL SAFE 
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'inference_shared_write_once_int_wrapper';

-- src/lib.rs:219
-- pg_extension::inference_shared_write_once
CREATE  FUNCTION "inference_shared_write_once"(
	"dataset" TEXT, /* alloc::string::String */
	"condition" TEXT, /* alloc::string::String */
	"config_file" TEXT, /* alloc::string::String */
	"col_cardinalities_file" TEXT, /* alloc::string::String */
	"model_path" TEXT, /* alloc::string::String */
	"sql" TEXT, /* alloc::string::String */
	"batch_size" INT /* i32 */
) RETURNS TEXT /* alloc::string::String */
IMMUTABLE STRICT PARALLEL SAFE 
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'inference_shared_write_once_wrapper';

-- src/lib.rs:196
-- pg_extension::inference_shared
CREATE  FUNCTION "inference_shared"(
	"dataset" TEXT, /* alloc::string::String */
	"condition" TEXT, /* alloc::string::String */
	"config_file" TEXT, /* alloc::string::String */
	"col_cardinalities_file" TEXT, /* alloc::string::String */
	"model_path" TEXT, /* alloc::string::String */
	"sql" TEXT, /* alloc::string::String */
	"batch_size" INT /* i32 */
) RETURNS TEXT /* alloc::string::String */
IMMUTABLE STRICT PARALLEL SAFE 
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'run_inference_shared_wrapper';

-- src/lib.rs:173
-- pg_extension::inference
CREATE  FUNCTION "inference"(
	"dataset" TEXT, /* alloc::string::String */
	"condition" TEXT, /* alloc::string::String */
	"config_file" TEXT, /* alloc::string::String */
	"col_cardinalities_file" TEXT, /* alloc::string::String */
	"model_path" TEXT, /* alloc::string::String */
	"sql" TEXT, /* alloc::string::String */
	"batch_size" INT /* i32 */
) RETURNS TEXT /* alloc::string::String */
IMMUTABLE STRICT PARALLEL SAFE 
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'run_inference_wrapper';


# record the necessary func above and then copy it to following
rm /home/postgres/.pgrx/14.9/pgrx-install/share/extension/pg_extension--0.1.0.sql
vi /home/postgres/.pgrx/14.9/pgrx-install/share/extension/pg_extension--0.1.0.sql

# then drop/create extension
DROP EXTENSION IF EXISTS pg_extension;
CREATE EXTENSION pg_extension;
```

Examples

```sql

# this is database name, columns used, time budget, batch size, and config file
SELECT count(*) FROM frappe_train WHERE col2='973:1' LIMIT 1000;
SELECT col2, count(*) FROM frappe_train group by col2 order by count(*) desc;

# query with two conditions
SELECT inference(
    'frappe', 
    '{"1":266, "2":1244}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    'WHERE col1=''266:1'' and col2=''1244:1''', 
    32
);

# query with 1 conditions
SELECT inference(
    'frappe', 
    '{"2":977}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    'WHERE col2=''977:1''', 
    10000
); 

# query with no conditions
SELECT inference(
    'frappe', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    8000
); 

# explaination
EXPLAIN (ANALYZE, BUFFERS) SELECT inference(
    'frappe', 
    '{"2":977}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    'WHERE col2=''977:1''', 
    8000
); 


```

# Clear cache

```sql
DISCARD ALL;
```

# Benchmark Latency over all datasets

## Adult

```sql
SELECT inference(
    'adult', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/data/adult_col_cardinalities', 
    '/project/tensor_log/adult/Ednn_K16_alpha2-5', 
    '', 
    10000
); 


# exps
SELECT model_init(
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/data/adult_col_cardinalities', 
    '/project/tensor_log/adult/Ednn_K16_alpha2-5', 
); 
SELECT inference(
    'adult', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/data/adult_col_cardinalities', 
    '/project/tensor_log/adult/Ednn_K16_alpha2-5', 
    '', 
    10000
); 


SELECT model_init(
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/data/adult_col_cardinalities', 
    '/project/tensor_log/adult/Ednn_K16_alpha2-5'
); 
SELECT inference_shared_write_once(
    'adult', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/data/adult_col_cardinalities', 
    '/project/tensor_log/adult/Ednn_K16_alpha2-5', 
    '', 
    100000
); 

# replicate data 
INSERT INTO adult_train (label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13)
SELECT label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13
FROM adult_train;

INSERT INTO adult_int_train (label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13)
SELECT label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13
FROM adult_int_train;
```

## Frappe

```sql
SELECT inference(
    'frappe', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/data/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    10000
); 

SELECT inference(
    'frappe', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/data/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    20000
); 


SELECT model_init(
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/data/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4'
); 
SELECT inference(
    'frappe', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/data/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    10000
); 



SELECT model_init(
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/data/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4'
); 
SELECT inference_shared_write_once(
    'frappe', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/data/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    100000
); 


SELECT inference_shared(
    'frappe', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/data/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    40000
); 



SELECT inference(
    'frappe', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/data/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    80000
); 


SELECT inference(
    'frappe', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/data/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    160000
); 

# replicate data 
INSERT INTO frappe_train (label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10)
SELECT label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10
FROM frappe_train;


INSERT INTO frappe_int_train (label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10)
SELECT label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10
FROM frappe_int_train;
```

## CVD

```sql
SELECT inference(
    'cvd', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/data/cvd_col_cardinalities', 
    '/project/tensor_log/cvd/dnn_K16_alpha2-5', 
    '', 
    10000
); 

# exps
SELECT model_init(
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/data/cvd_col_cardinalities', 
    '/project/tensor_log/cvd/dnn_K16_alpha2-5', 
); 
SELECT inference(
    'cvd', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/data/cvd_col_cardinalities', 
    '/project/tensor_log/cvd/dnn_K16_alpha2-5', 
    '', 
    10000
); 


SELECT model_init(
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/data/cvd_col_cardinalities', 
    '/project/tensor_log/cvd/dnn_K16_alpha2-5'
); 
SELECT inference_shared_write_once(
    'cvd', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/data/cvd_col_cardinalities', 
    '/project/tensor_log/cvd/dnn_K16_alpha2-5', 
    '', 
    100000
); 


# replicate data 
INSERT INTO cvd_train (label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11)
SELECT label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 
FROM cvd_train;

INSERT INTO cvd_int_train (label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11)
SELECT label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11 
FROM cvd_int_train;

```

## Bank

```sql
SELECT inference(
    'bank', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/data/bank_col_cardinalities', 
    '/project/tensor_log/bank/dnn_K16_alpha2-3_beta1e-3', 
    '', 
    10000
); 


# exps
SELECT model_init(
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/data/bank_col_cardinalities', 
    '/project/tensor_log/bank/dnn_K16_alpha2-3_beta1e-3', 
); 
SELECT inference(
    'bank', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/data/bank_col_cardinalities', 
    '/project/tensor_log/bank/dnn_K16_alpha2-3_beta1e-3', 
    '', 
    10000
); 


SELECT model_init(
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/bank_col_cardinalities', 
    '/project/tensor_log/bank/dnn_K16_alpha2-3_beta1e-3'
); 
SELECT inference_shared_write_once(
    'bank', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/bank_col_cardinalities', 
    '/project/tensor_log/bank/dnn_K16_alpha2-3_beta1e-3', 
    '', 
    100000
); 


# replicate data 
INSERT INTO bank_train (label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16)
SELECT label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16 
FROM bank_train;


INSERT INTO bank_int_train (label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16)
SELECT label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16 
FROM bank_int_train;

```

## Census

```sql
# replicate data 
INSERT INTO census_int_train (label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27,col28,col29,col30,col31,col32,col33,col34,col35,col36,col37,col38,col39,col40,col41)
SELECT label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27,col28,col29,col30,col31,col32,col33,col34,col35,col36,col37,col38,col39,col40,col41
FROM census_int_train;


```

## Credit

```sql
# replicate data 
INSERT INTO credit_int_train (label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16,col17,col18,col19,col20,col21,col22,col23)
SELECT label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16,col17,col18,col19,col20,col21,col22,col23
FROM credit_int_train;



```

## Diabetes

```sql
# replicate data 
INSERT INTO diabetes_int_train (label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27,col28,col29,col30,col31,col32,col33,col34,col35,col36,col37,col38,col39,col40,col41,col42,col43,col44,col45,col46,col47,col48)
SELECT label, col1, col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14, col15, col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27,col28,col29,col30,col31,col32,col33,col34,col35,col36,col37,col38,col39,col40,col41,col42,col43,col44,col45,col46,col47,col48
FROM diabetes_int_train;



```

## Hcdr

```sql
# replicate data 
INSERT INTO hcdr_int_train (label,col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27,col28,col29,col30,col31,col32,col33,col34,col35,col36,col37,col38,col39,col40,col41,col42,col43,col44,col45,col46,col47,col48,col49,col50,col51,col52,col53,col54,col55,col56,col57,col58,col59,col60,col61,col62,col63,col64,col65,col66,col67,col68,col69)
SELECT label,col1,col2,col3,col4,col5,col6,col7,col8,col9,col10,col11,col12,col13,col14,col15,col16,col17,col18,col19,col20,col21,col22,col23,col24,col25,col26,col27,col28,col29,col30,col31,col32,col33,col34,col35,col36,col37,col38,col39,col40,col41,col42,col43,col44,col45,col46,col47,col48,col49,col50,col51,col52,col53,col54,col55,col56,col57,col58,col59,col60,col61,col62,col63,col64,col65,col66,col67,col68,col69
FROM hcdr_int_train;


```



# Baseline System & SAMS

## Frappe

```bash
# frappe
CUDA_VISIBLE_DEVICES=-1 python ./internal/ml/model_slicing/algorithm/baseline.py /hdd1/sams/tensor_log/frappe/dnn_K16_alpha4 --device cpu --dataset frappe --batch_size 10 --col_cardinalities_file data/frappe_col_cardinalities --target_batch 10


CUDA_VISIBLE_DEVICES="0" python ./internal/ml/model_slicing/algorithm/baseline.py /hdd1/sams/tensor_log/frappe/dnn_K16_alpha4 --device cuda:0 --dataset frappe --batch_size 100000 --col_cardinalities_file data/frappe_col_cardinalities --target_batch 100000

CUDA_VISIBLE_DEVICES=-1 python ./internal/ml/model_slicing/algorithm/baseline_int.py /hdd1/sams/tensor_log/frappe/dnn_K16_alpha4 --device cpu --dataset frappe --batch_size 100000 --col_cardinalities_file data/frappe_col_cardinalities --target_batch 100000


SELECT model_init(
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4'
); 
SELECT inference_shared_write_once(
    'frappe', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    100000
); 

# read int data
SELECT model_init(
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4'
); 
SELECT inference_shared_write_once_int(
    'frappe', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    100000
); 
```

## Adult

```bash

# adult
CUDA_VISIBLE_DEVICES=-1 python ./internal/ml/model_slicing/algorithm/baseline.py /hdd1/sams/tensor_log/adult/Ednn_K16_alpha2-5 --device cpu --dataset adult --batch_size 100000 --col_cardinalities_file data/adult_col_cardinalities  --target_batch 100000

CUDA_VISIBLE_DEVICES="0" python ./internal/ml/model_slicing/algorithm/baseline.py /hdd1/sams/tensor_log/adult/Ednn_K16_alpha2-5 --device cuda:0 --dataset adult --batch_size 100000 --col_cardinalities_file data/adult_col_cardinalities  --target_batch 100000

CUDA_VISIBLE_DEVICES=-1 python ./internal/ml/model_slicing/algorithm/baseline_int.py /hdd1/sams/tensor_log/adult/Ednn_K16_alpha2-5 --device cpu --dataset adult --batch_size 100000 --col_cardinalities_file data/adult_col_cardinalities  --target_batch 100000

SELECT model_init(
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/adult_col_cardinalities', 
    '/project/tensor_log/adult/Ednn_K16_alpha2-5'
); 
SELECT inference_shared_write_once(
    'adult', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/adult_col_cardinalities', 
    '/project/tensor_log/adult/Ednn_K16_alpha2-5', 
    '', 
    100000
); 

SELECT model_init(
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/adult_col_cardinalities', 
    '/project/tensor_log/adult/Ednn_K16_alpha2-5'
); 
SELECT inference_shared_write_once_int(
    'adult', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/adult_col_cardinalities', 
    '/project/tensor_log/adult/Ednn_K16_alpha2-5', 
    '', 
    640000
); 
```

## CVD
```bash
# CVD
CUDA_VISIBLE_DEVICES=-1 python ./internal/ml/model_slicing/algorithm/baseline.py /hdd1/sams/tensor_log/cvd/dnn_K16_alpha2-5 --device cpu --dataset cvd --batch_size 100000 --col_cardinalities_file data/cvd_col_cardinalities  --target_batch 100000

CUDA_VISIBLE_DEVICES="0" python ./internal/ml/model_slicing/algorithm/baseline.py /hdd1/sams/tensor_log/cvd/dnn_K16_alpha2-5 --device cuda:0 --dataset cvd --batch_size 100000 --col_cardinalities_file data/cvd_col_cardinalities  --target_batch 100000

CUDA_VISIBLE_DEVICES=-1 python ./internal/ml/model_slicing/algorithm/baseline_int.py /hdd1/sams/tensor_log/cvd/dnn_K16_alpha2-5 --device cpu --dataset cvd --batch_size 100000 --col_cardinalities_file data/cvd_col_cardinalities  --target_batch 100000


SELECT model_init(
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/cvd_col_cardinalities', 
    '/project/tensor_log/cvd/dnn_K16_alpha2-5'
); 
SELECT inference_shared_write_once(
    'cvd', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/cvd_col_cardinalities', 
    '/project/tensor_log/cvd/dnn_K16_alpha2-5', 
    '', 
    100000
); 

SELECT model_init(
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/cvd_col_cardinalities', 
    '/project/tensor_log/cvd/dnn_K16_alpha2-5'
); 
SELECT inference_shared_write_once_int(
    'cvd', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/cvd_col_cardinalities', 
    '/project/tensor_log/cvd/dnn_K16_alpha2-5', 
    '', 
    100000
); 
```

## Bank

```bash
# Bank
CUDA_VISIBLE_DEVICES=-1 python ./internal/ml/model_slicing/algorithm/baseline.py /hdd1/sams/tensor_log/bank/dnn_K16_alpha2-3_beta1e-3 --device cpu --dataset bank --batch_size 100000 --col_cardinalities_file data/bank_col_cardinalities  --target_batch 100000

CUDA_VISIBLE_DEVICES="0" python ./internal/ml/model_slicing/algorithm/baseline.py /hdd1/sams/tensor_log/bank/dnn_K16_alpha2-3_beta1e-3 --device cuda:0 --dataset bank --batch_size 100000 --col_cardinalities_file data/bank_col_cardinalities  --target_batch 100000


CUDA_VISIBLE_DEVICES=-1 python ./internal/ml/model_slicing/algorithm/baseline_int.py /hdd1/sams/tensor_log/bank/dnn_K16_alpha2-3_beta1e-3 --device cpu --dataset bank --batch_size 100000 --col_cardinalities_file data/bank_col_cardinalities  --target_batch 100000

SELECT model_init(
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/bank_col_cardinalities', 
    '/project/tensor_log/bank/dnn_K16_alpha2-3_beta1e-3'
); 
SELECT inference_shared_write_once(
    'bank', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/bank_col_cardinalities', 
    '/project/tensor_log/bank/dnn_K16_alpha2-3_beta1e-3', 
    '', 
    100000
); 


SELECT model_init(
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/bank_col_cardinalities', 
    '/project/tensor_log/bank/dnn_K16_alpha2-3_beta1e-3'
); 
SELECT inference_shared_write_once_int(
    'bank', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/bank_col_cardinalities', 
    '/project/tensor_log/bank/dnn_K16_alpha2-3_beta1e-3', 
    '', 
    100000
); 


```

## Census

```sql
# Bank
CUDA_VISIBLE_DEVICES=-1 python ./internal/ml/model_slicing/baseline_int.py /hdd1/sams/tensor_log/census/dnn_K16 --device cpu --dataset census --batch_size 100000 --col_cardinalities_file ./internal/ml/model_slicing/data/census_col_cardinalities  --target_batch 100000

SELECT model_init(
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/internal/ml/model_slicing/data/census_col_cardinalities', 
    '/project/tensor_log/census/dnn_K16'
); 
SELECT inference_shared_write_once_int(
    'census', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/internal/ml/model_slicing/data/census_col_cardinalities', 
    '/project/tensor_log/census/dnn_K16', 
    '', 
    100000
); 
```

## Credit

```sql
# Bank
CUDA_VISIBLE_DEVICES=-1 python ./internal/ml/model_slicing/baseline_int.py /hdd1/sams/tensor_log/credit/dnn_K16_epoch50 --device cpu --dataset credit --batch_size 100000 --col_cardinalities_file ./internal/ml/model_slicing/data/credit_col_cardinalities  --target_batch 100000

SELECT model_init(
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/internal/ml/model_slicing/data/credit_col_cardinalities', 
    '/project/tensor_log/credit/dnn_K16_epoch50'
); 
SELECT inference_shared_write_once_int(
    'credit', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/internal/ml/model_slicing/data/credit_col_cardinalities', 
    '/project/tensor_log/credit/dnn_K16_epoch50', 
    '', 
    100000
); 
```

## Diabetes

```sql
# Bank
CUDA_VISIBLE_DEVICES=-1 python ./internal/ml/model_slicing/baseline_int.py /hdd1/sams/tensor_log/diabetes/dnn_K16_epoch50 --device cpu --dataset diabetes --batch_size 100000 --col_cardinalities_file ./internal/ml/model_slicing/data/diabetes_col_cardinalities  --target_batch 100000

SELECT model_init(
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/internal/ml/model_slicing/data/diabetes_col_cardinalities', 
    '/project/tensor_log/diabetes/dnn_K16_epoch50'
); 
SELECT inference_shared_write_once_int(
    'diabetes', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/internal/ml/model_slicing/data/diabetes_col_cardinalities', 
    '/project/tensor_log/diabetes/dnn_K16_epoch50', 
    '', 
    100000
); 
```

## Hcdr

```sql
# Bank
CUDA_VISIBLE_DEVICES=-1 python ./internal/ml/model_slicing/baseline_int.py /hdd1/sams/tensor_log/hcdr/dnn_K16 --device cpu --dataset hcdr --batch_size 100000 --col_cardinalities_file ./internal/ml/model_slicing/data/hcdr_col_cardinalities  --target_batch 100000

SELECT model_init(
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/internal/ml/model_slicing/data/hcdr_col_cardinalities', 
    '/project/tensor_log/hcdr/dnn_K16'
); 
SELECT inference_shared_write_once_int(
    'hcdr', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/internal/ml/model_slicing/data/hcdr_col_cardinalities', 
    '/project/tensor_log/hcdr/dnn_K16', 
    '', 
    100000
); 
```

# Data Scale

```sql
# Bank
CUDA_VISIBLE_DEVICES=-1 python ./internal/ml/model_slicing/baseline_int.py /hdd1/sams/tensor_log/credit/dnn_K16_epoch50 --device cpu --dataset credit --batch_size 640000 --col_cardinalities_file ./internal/ml/model_slicing/data/credit_col_cardinalities  --target_batch 640000

SELECT model_init(
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/internal/ml/model_slicing/data/credit_col_cardinalities', 
    '/project/tensor_log/credit/dnn_K16_epoch50'
); 
SELECT inference_shared_write_once_int(
    'credit', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/internal/ml/model_slicing/data/credit_col_cardinalities', 
    '/project/tensor_log/credit/dnn_K16_epoch50', 
    '', 
    640000
); 
```

# Micro

## Profiling

```bash
CUDA_VISIBLE_DEVICES=-1 python ./internal/ml/model_slicing/algorithm/baseline.py /hdd1/sams/tensor_log/frappe/dnn_K16_alpha4 --device cpu --dataset frappe --batch_size 20000 --col_cardinalities_file frappe_col_cardinalities --target_batch 20000`
```

## Optimizations

```bash

# 1. with all opt
SELECT model_init(
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4'
); 
SELECT inference_shared_write_once(
    'frappe', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    100000
); 

# 2. w/o model cache
SELECT inference_shared_write_once(
    'frappe', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    100000
); 

# 3. w/o shared memory
SELECT model_init(
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4'
); 
SELECT inference(
    'frappe', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/frappe_col_cardinalities', 
    '/project/tensor_log/frappe/dnn_K16_alpha4', 
    '', 
    100000
); 

# w/o SPI this can measure the time usage for not using spi
CUDA_VISIBLE_DEVICES=-1 python ./internal/ml/model_slicing/algorithm/baseline.py /hdd1/sams/tensor_log/frappe/dnn_K16_alpha4 --device cpu --dataset frappe --batch_size 100000 --col_cardinalities_file frappe_col_cardinalities --target_batch 100000
```

Int dataset

```bash

# 1. with all opt
SELECT model_init(
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/internal/ml/model_slicing/data/credit_col_cardinalities', 
    '/project/tensor_log/credit/dnn_K16_epoch50'
); 
SELECT inference_shared_write_once_int(
    'credit', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/internal/ml/model_slicing/data/credit_col_cardinalities', 
    '/project/tensor_log/credit/dnn_K16_epoch50', 
    '', 
    100000
); 

# 2. w/o model cache
SELECT inference_shared_write_once_int(
    'credit', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/internal/ml/model_slicing/data/credit_col_cardinalities', 
    '/project/tensor_log/credit/dnn_K16_epoch50', 
    '', 
    100000
); 

# 3. w/o shared memory
SELECT model_init(
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/internal/ml/model_slicing/data/credit_col_cardinalities', 
    '/project/tensor_log/credit/dnn_K16_epoch50'
); 
SELECT inference(
    'credit', 
    '{}', 
    '/project/Trails/internal/ml/model_selection/config.ini', 
    '/project/Trails/internal/ml/model_slicing/data/credit_col_cardinalities', 
    '/project/tensor_log/credit/dnn_K16_epoch50', 
    '', 
    100000
); 

# w/o SPI this can measure the time usage for not using spi
CUDA_VISIBLE_DEVICES=-1 python ./internal/ml/model_slicing/algorithm/baseline.py /hdd1/sams/tensor_log/frappe/dnn_K16_alpha4 --device cpu --dataset frappe --batch_size 100000 --col_cardinalities_file frappe_col_cardinalities --target_batch 100000

CUDA_VISIBLE_DEVICES=-1 python ./internal/ml/model_slicing/baseline_int.py /hdd1/sams/tensor_log/credit/dnn_K16_epoch50 --device cpu --dataset credit --batch_size 100000 --col_cardinalities_file ./internal/ml/model_slicing/data/credit_col_cardinalities  --target_batch 100000
```













