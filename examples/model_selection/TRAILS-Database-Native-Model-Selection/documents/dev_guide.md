



# Change the permission

```bash
chmod -R 777 internal/pg_extension
chmod -R 777 TRAILS
```

# PSQL CMD

```sql
psql -h localhost -p 28814 -U postgres 
\c frappe
\dt
\d frappe_train
DROP TABLE frappe_train;
SELECT * FROM frappe_train LIMIT 10;
SELECT * FROM frappe_test LIMIT 10;
SELECT * FROM frappe_valid LIMIT 10;
DROP DATABASE frappe;
psql -U postgres
```

# Build and run the container

```bash
docker build -t trails .

docker run -d --name trails \
  --network="host" \
  -v $(pwd)/TRAILS:/project/TRAILS \
  -v /hdd1/xingnaili/exp_data/:/project/exp_data \
  trails

docker exec -it trails bash 
```

# This is in docker image already

```bash
# if those are already on docker, skip them.
cargo install --locked cargo-pgrx
# run after package update
cargo pgrx init
cargo pgrx new my_extension
# just run this after code updates.
cargo pgrx run
```

# Develop

## Load data into database.

```bash
bash /project/TRAILS/internal/ml/model_selection/scripts/database/load_data_to_db.sh /project/exp_data/data/structure_data/frappe frappe
bash /project/TRAILS/internal/ml/model_selection/scripts/database/load_data_to_db.sh /project/exp_data/data/structure_data/uci_diabetes uci_diabetes
bash /project/TRAILS/internal/ml/model_selection/scripts/database/load_data_to_db.sh /project/exp_data/data/structure_data/criteo_full criteo
```

## 1. Compile

In shell

```bash
cd ./internal/pg_extension/
cargo clean
rm -r /home/postgres/.pgrx/14.9/pgrx-install/lib/pg_extension.so
cargo pgrx run
rm /home/postgres/.pgrx/14.9/pgrx-install/share/extension/pg_extension--0.1.0.sql
vi /home/postgres/.pgrx/14.9/pgrx-install/share/extension/pg_extension--0.1.0.sql
paste the latest sqls
# generate schema
cargo pgrx schema >> /home/postgres/.pgrx/14.9/pgrx-install/share/extension/pg_extension--0.1.0.sql
```

In SQL

```sql
DROP EXTENSION IF EXISTS pg_extension;
CREATE EXTENSION pg_extension;
```

## 2. Edit the config file

Update the `nfield` in the `config.ini` file, it is == number of columns used. E.g, `ARRAY['col1', 'col2', 'col3', 'label']`  => `nfield` = 3

## 3. Run it

```sql
CREATE EXTENSION pg_extension;

# Test if the UDF is there or not
SELECT *  FROM pg_proc  WHERE proname = 'model_selection_workloads';

# micro
select benchmark_filtering_phase_latency(4, '/project/TRAILS/internal/ml/model_selection/config.ini');

select benchmark_filtering_latency_in_db(5000, 'frappe', '/project/TRAILS/internal/ml/model_selection/config.ini');

select benchmark_filtering_latency_in_db(5000, 'uci_diabetes', '/project/TRAILS/internal/ml/model_selection/config.ini');

select benchmark_filtering_latency_in_db(4, 'criteo', '/project/TRAILS/internal/ml/model_selection/config.ini');

# Test coordinator
SELECT coordinator('0.08244', '168.830156', '800', false, '/project/TRAILS/internal/ml/model_selection/config.ini');

# this is database name, columns used, time budget, batch size, and config file
CALL model_selection_sp('dummy', ARRAY['col1', 'col2', 'col3', 'label'], '30', 32, '/project/TRAILS/internal/ml/model_selection/config.ini');

# end2end model selection
CALL model_selection_end2end('dummy', ARRAY['col1', 'col2', 'col3', 'label'], '15', '/project/TRAILS/internal/ml/model_selection/config.ini');

# filtering & refinement with workloads
CALL model_selection_workloads('dummy', ARRAY['col1', 'col2', 'col3', 'label'], 300, 3, '/project/TRAILS/internal/ml/model_selection/config.ini');

response = requests.post(args.refinement_url, json=data).json()

```

# Test the pg-extension works using pipython

```sql
# switch to a postgres
su postgres

CREATE EXTENSION plpython3u;

CREATE FUNCTION py_version() RETURNS text AS $$
import sys
return sys.version
$$ LANGUAGE plpython3u;

SELECT py_version();

CREATE OR REPLACE FUNCTION test_numpy()
  RETURNS text
LANGUAGE plpython3u
AS $$
import numpy
import torch
import sklearn
import torchvision
import tqdm
print("asdf")
return str(numpy.__version__) + " torch: " + str(torch.__version__)
$$;

SELECT test_numpy();

CREATE EXTENSION my_extension;
SELECT hello_my_extension();
```

# Container log

Each line in your output represents a different process that is currently running on your PostgreSQL server. Here's what each one is doing:

1.  `/bin/sh -c service postgresql start && tail -F /var/log/postgresql/postgresq` : This is the command that was used to start your PostgreSQL server. It also includes a command to continuously display new entries from the PostgreSQL log file.


2.  `/usr/lib/postgresql/14/bin/postgres -D /var/lib/postgresql/14/main -c config` : This is the main PostgreSQL process. All other PostgreSQL processes are children of this process.


3.  `postgres: 14/main: checkpointer` : The checkpointer process is responsible for making sure data changes get saved to disk regularly. This is important for database recovery in case of a crash.


4.  `postgres: 14/main: background writer` : The background writer process is responsible for writing buffers to disk when they become dirty. This reduces the amount of work that needs to be done when a buffer is reused.


5.  `postgres: 14/main: walwriter` : The walwriter process writes transaction logs (Write-Ahead Logs or WAL) to disk. This is also important for database recovery and replication.


6.  `postgres: 14/main: autovacuum launcher` : The autovacuum launcher process starts autovacuum worker processes as needed. These processes automatically clean up and optimize the database.


7.  `postgres: 14/main: stats collector` : The stats collector process collects statistics about the server's activity. This information can be viewed using the `pg_stat` family of system views.


8.  `postgres: 14/main: logical replication launcher` : The logical replication launcher manages the worker processes that perform logical replication, copying data changes to other databases.


9.  `tail -F /var/log/postgresql/postgresql-14-main.log` : This process is displaying the end of the PostgreSQL log file and updating as more entries are added.


10.  `bash` : These are shell sessions, likely interactive ones you've started.


11.  `/usr/lib/postgresql/14/bin/psql -h localhost -p 28814 pg_extension` : These are instances of the psql command line interface, connected to your database.


12.  `postgres: postgres pg_extension 127.0.0.1(52236) CALL` : This is your currently running stored procedure.


13.  `ps aux` : This is the command you ran to display the list of processes.

Each process is part of the PostgreSQL database system and helps it to run efficiently and robustly.

# MAC locally

```bash
conda activate firmest38
export PYTHON_SYS_EXECUTABLE=/Users/kevin/opt/anaconda3/envs/firmest38/bin/python
export DYLD_LIBRARY_PATH=/Users/kevin/opt/anaconda3/envs/firmest38/lib/:$DYLD_LIBRARY_PATH
cargo run --features python
```

# What cargo run do?

Before:

```
postgres     1  0.1  0.0   2612   588 ?        Ss   14:30   0:00 /bin/sh -c service postgresql start && tail -F /var/log/postgresql/postgresql-14-main.log
postgres    20  0.1  0.0 214688 29332 ?        Ss   14:30   0:00 /usr/lib/postgresql/14/bin/postgres -D /var/lib/postgresql/14/main -c config_file=/etc/postgresql/14/main/postgresql.conf
postgres    22  0.0  0.0 214688  6120 ?        Ss   14:30   0:00 postgres: 14/main: checkpointer 
postgres    23  0.0  0.0 214688  6084 ?        Ss   14:30   0:00 postgres: 14/main: background writer 
postgres    24  0.0  0.0 214688 10352 ?        Ss   14:30   0:00 postgres: 14/main: walwriter 
postgres    25  0.0  0.0 215224  8864 ?        Ss   14:30   0:00 postgres: 14/main: autovacuum launcher 
postgres    26  0.0  0.0  69280  5184 ?        Ss   14:30   0:00 postgres: 14/main: stats collector 
postgres    27  0.0  0.0 215236  6972 ?        Ss   14:30   0:00 postgres: 14/main: logical replication launcher 
postgres    38  0.0  0.0   2548   512 ?        S    14:30   0:00 tail -F /var/log/postgresql/postgresql-14-main.log
postgres    39  0.1  0.0   4112  3424 pts/0    Ss+  14:30   0:00 bash
postgres    48  0.1  0.0   4112  3424 pts/1    Ss   14:30   0:00 bash
postgres    59  0.0  0.0   5896  2860 pts/1    R+   14:30   0:00 ps aux
```

After:







