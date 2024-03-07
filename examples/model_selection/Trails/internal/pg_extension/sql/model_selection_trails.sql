/************************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/


CREATE OR REPLACE
PROCEDURE model_selection_end2end(
    dataset TEXT,               --dataset name
    selected_columns TEXT[],    --used columns
    budget TEXT,                --user given time budget
    config_file TEXT            --config file path
)
LANGUAGE plpgsql
AS $$
DECLARE
    -- global inputs/outputs
    result_status TEXT;
    column_list TEXT;

BEGIN
    -- combine the columns into a string
    column_list := array_to_string(selected_columns, ', ');
    EXECUTE format('
                WITH batch_rows AS (
                    SELECT %s
                    FROM %I
                    ORDER BY RANDOM()
                )
                SELECT model_selection_trails(
                    json_agg(row_to_json(t))::text, %L, %L
                )
                FROM batch_rows AS t', column_list, dataset, budget, config_file) INTO result_status;
    RAISE NOTICE '1. model_selection result: %', result_status;
END; $$;
