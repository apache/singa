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

# Database-Native Model Selection 

​																																																		-- based on SINGA



![image-20231020174425377](documents/image-20231020174425377.png)


# Build & Run examples

## Singa + PostgreSQL

```bash
# Create project folder.
mkdir project && cd project
# Download the Dockerile.
wget https://raw.githubusercontent.com/apache/singa/dev-postgresql/examples/model_selection/TRAILS-Database-Native-Model-Selection/Dockerfile


# Build Dockerile and run the docker.
docker build -t trails .
docker run -d --name trails --network="host" trails
# Monitor the logs until the setup step is done.
docker logs -f trails

docker exec -it trails bash
# Connect to the pg server and use pg_extension database.
psql -h localhost -p 28814 -U postgres
\c pg_extension

# Run an example, wait one min, it will run filtering + refinemnt + training the selected model.
CALL model_selection_end2end('frappe_train', ARRAY['col1', 'col2', 'col3', 'col4','col5','col6','col7','col8','col9','col10', 'label'], '10', '/project/Trails/internal/ml/model_selection/config.ini');

```


## Singa + PolarDB

```bash
# Create project folder.
mkdir project && cd project
# Download the Dockerile.
wget https://raw.githubusercontent.com/apache/singa/dev-postgresql/examples/model_selection/TRAILS-Database-Native-Model-Selection/singa.polarDB.Dockerfile


# Build Dockerile and run the docker.
docker build -t trails_polardb .
docker run -d --name trails_polardb --network="host" trails_polardb
# Monitor the logs until the setup step is done.
docker logs -f trails_polardb

docker exec -it trails_polardb bash
# Connect to the pg server and use pg_extension database.
psql -h localhost -p 5432 -U postgres 
\c pg_extension

# Run an example, wait one min, it will run filtering + refinemnt + training the selected model.
CALL model_selection_end2end('frappe_train', ARRAY['col1', 'col2', 'col3', 'col4','col5','col6','col7','col8','col9','col10', 'label'], '10', '/project/Trails/internal/ml/model_selection/config.ini');

```

