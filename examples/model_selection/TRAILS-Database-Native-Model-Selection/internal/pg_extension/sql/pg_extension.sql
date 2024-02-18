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

-- src/lib.rs:66
-- pg_extension::filtering_phase
CREATE  FUNCTION "filtering_phase"(
    "mini_batch" TEXT, /* alloc::string::String */
    "n" INT, /* i32 */
    "k" INT, /* i32 */
    "config_file" TEXT /* alloc::string::String */
) RETURNS TEXT /* alloc::string::String */
    IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'filtering_phase_wrapper';

-- src/lib.rs:16
-- pg_extension::profiling_filtering_phase
CREATE  FUNCTION "profiling_filtering_phase"(
    "mini_batch" TEXT, /* alloc::string::String */
    "config_file" TEXT /* alloc::string::String */
) RETURNS TEXT /* alloc::string::String */
    IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'profiling_filtering_phase_wrapper';

-- src/lib.rs:80
-- pg_extension::refinement_phase
CREATE  FUNCTION "refinement_phase"(
    "config_file" TEXT /* alloc::string::String */
) RETURNS TEXT /* alloc::string::String */
    IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'refinement_phase_wrapper';

-- src/lib.rs:31
-- pg_extension::profiling_refinement_phase
CREATE  FUNCTION "profiling_refinement_phase"(
    "mini_batch" TEXT, /* alloc::string::String */
    "config_file" TEXT /* alloc::string::String */
) RETURNS TEXT /* alloc::string::String */
    IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'profiling_refinement_phase_wrapper';

-- src/lib.rs:46
-- pg_extension::coordinator
CREATE  FUNCTION "coordinator"(
    "time_score" TEXT, /* alloc::string::String */
    "time_train" TEXT, /* alloc::string::String */
    "time_budget" TEXT, /* alloc::string::String */
    "only_phase1" bool, /* bool */
    "config_file" TEXT /* alloc::string::String */
) RETURNS TEXT /* alloc::string::String */
    IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'coordinator_wrapper';

-- src/lib.rs:94
-- pg_extension::model_selection
CREATE  FUNCTION "model_selection"(
    "mini_batch" TEXT, /* alloc::string::String */
    "time_budget" TEXT, /* alloc::string::String */
    "config_file" TEXT /* alloc::string::String */
) RETURNS TEXT /* alloc::string::String */
    IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'model_selection_wrapper';

-- src/lib.rs:110
-- pg_extension::model_selection_workloads
CREATE  FUNCTION "model_selection_workloads"(
    "mini_batch" TEXT, /* alloc::string::String */
    "n" INT, /* i32 */
    "k" INT, /* i32 */
    "config_file" TEXT /* alloc::string::String */
) RETURNS TEXT /* alloc::string::String */
    IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'model_selection_workloads_wrapper';

-- src/lib.rs:125
-- pg_extension::model_selection_trails
CREATE  FUNCTION "model_selection_trails"(
    "mini_batch" TEXT, /* alloc::string::String */
    "time_budget" TEXT, /* alloc::string::String */
    "config_file" TEXT /* alloc::string::String */
) RETURNS TEXT /* alloc::string::String */
    IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'model_selection_trails_wrapper';

-- src/lib.rs:138
-- pg_extension::model_selection_trails_workloads
CREATE  FUNCTION "model_selection_trails_workloads"(
    "mini_batch" TEXT, /* alloc::string::String */
    "n" INT, /* i32 */
    "k" INT, /* i32 */
    "config_file" TEXT /* alloc::string::String */
) RETURNS TEXT /* alloc::string::String */
    IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'model_selection_trails_workloads_wrapper';

-- src/lib.rs:152
-- pg_extension::benchmark_filtering_phase_latency
CREATE  FUNCTION "benchmark_filtering_phase_latency"(
    "explore_models" INT, /* i32 */
    "config_file" TEXT /* alloc::string::String */
) RETURNS TEXT /* alloc::string::String */
    IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'benchmark_filtering_phase_latency_wrapper';

-- src/lib.rs:163
-- pg_extension::benchmark_filtering_latency_in_db
CREATE  FUNCTION "benchmark_filtering_latency_in_db"(
    "explore_models" INT, /* i32 */
    "dataset" TEXT, /* alloc::string::String */
    "config_file" TEXT /* alloc::string::String */
) RETURNS TEXT /* alloc::string::String */
    IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'benchmark_filtering_latency_in_db_wrapper';