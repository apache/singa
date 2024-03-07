#!/bin/bash
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



# Wait for PostgreSQL to become available
until psql -h localhost -p 5432 -U postgres -c '\q'; do
  >&2 echo "Postgres is unavailable - sleeping"
  sleep 1
done

# Run setup commands
psql -h localhost -p 5432 -U postgres -c "CREATE DATABASE pg_extension;"
psql -h localhost -p 5432 -U postgres -d pg_extension -c "CREATE EXTENSION pg_extension;"
psql -h localhost -p 5432 -U postgres -d pg_extension -f /home/postgres/Trails/internal/pg_extension/sql/model_selection_cpu.sql
# Load example dataset into database
bash /home/postgres/Trails/internal/ml/model_selection/scripts/database/load_data_to_db.sh //home/postgres/Trails/dataset/frappe frappe 5432

echo "Done!"
