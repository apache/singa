-- src/lib.rs:16
-- pg_extension::profiling_filtering_phase
CREATE  FUNCTION "profiling_filtering_phase"(
    "mini_batch" TEXT, /* alloc::string::String */
    "config_file" TEXT /* alloc::string::String */
) RETURNS TEXT /* alloc::string::String */
    IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'profiling_filtering_phase_wrapper';

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

-- src/lib.rs:153
-- pg_extension::benchmark_filtering_phase_latency
CREATE  FUNCTION "benchmark_filtering_phase_latency"(
    "explore_models" INT, /* i32 */
    "config_file" TEXT /* alloc::string::String */
) RETURNS TEXT /* alloc::string::String */
    IMMUTABLE STRICT PARALLEL SAFE
LANGUAGE c /* Rust */
AS 'MODULE_PATHNAME', 'benchmark_filtering_phase_latency_wrapper';


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