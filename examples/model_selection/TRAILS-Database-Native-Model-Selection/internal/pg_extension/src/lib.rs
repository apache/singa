use pgrx::prelude::*;
pgrx::pg_module_magic!();
use serde_json::json;
use std::collections::HashMap;

pub mod bindings;
extern crate serde_derive;

/*
 * @param mini_batch: mini_batch of data. Assume all columns are string type in
 * libsvm codding
 */
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "profiling_filtering_phase")]
#[allow(unused_variables)]
pub fn profiling_filtering_phase(mini_batch: String, config_file: String) -> String {
    let mut task_map = HashMap::new();
    task_map.insert("mini_batch", mini_batch);
    task_map.insert("config_file", config_file);
    let task_json = json!(task_map).to_string();
    crate::bindings::ms::profiling_filtering_phase(&task_json).to_string()
}

/*
 * @param mini_batch: training for one iteration.
 * libsvm codding
 */
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "profiling_refinement_phase")]
#[allow(unused_variables)]
pub fn profiling_refinement_phase(mini_batch: String, config_file: String) -> String {
    let mut task_map = HashMap::new();
    task_map.insert("mini_batch", mini_batch);
    task_map.insert("config_file", config_file);
    let task_json = json!(task_map).to_string();
    crate::bindings::ms::profiling_refinement_phase(&task_json).to_string()
}

/*
 * @param mini_batch: training for one iteration.
 * libsvm codding
 */
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "coordinator")]
#[allow(unused_variables)]
pub fn coordinator(time_score: String, time_train: String, time_budget: String, only_phase1: bool,
                   config_file: String) -> String {
    let mut task_map = HashMap::new();
    task_map.insert("budget", time_budget);
    task_map.insert("score_time_per_model", time_score);
    task_map.insert("train_time_per_epoch", time_train);
    task_map.insert("only_phase1", only_phase1.to_string());
    task_map.insert("config_file", config_file);
    let task_json = json!(task_map).to_string();
    crate::bindings::ms::coordinator(&task_json).to_string()
}


/*
 * @param mini_batch: mini_batch of data. Assume all columns are string type in
 * libsvm codding
 */
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "filtering_phase")]
#[allow(unused_variables)]
pub fn filtering_phase(mini_batch: String, n: i32, k: i32, config_file: String) -> String {
    let mut task_map = HashMap::new();
    task_map.insert("mini_batch", mini_batch);
    task_map.insert("n", n.to_string());
    task_map.insert("k", k.to_string());
    task_map.insert("config_file", config_file);
    let task_json = json!(task_map).to_string();
    crate::bindings::ms::filtering_phase(&task_json).to_string()
}


#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "refinement_phase")]
#[allow(unused_variables)]
pub fn refinement_phase(config_file: String) -> String {
    let mut task_map = HashMap::new();
    task_map.insert("config_file", config_file);
    let task_json = json!(task_map).to_string();
    crate::bindings::ms::refinement_phase().to_string()
}


/*
 End-2-End model selection, All in UDF runtime.
 */
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "model_selection")]
#[allow(unused_variables)]
pub fn model_selection(mini_batch: String, time_budget: String, config_file: String) -> String {
    let mut task_map = HashMap::new();
    task_map.insert("mini_batch", mini_batch);
    task_map.insert("budget", time_budget);
    task_map.insert("config_file", config_file);
    let task_json = json!(task_map).to_string();
    crate::bindings::ms::model_selection(&task_json).to_string()
}

/*
 * @param mini_batch: mini_batch of data. Assume all columns are string type in
 * libsvm codding
 */
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "model_selection_workloads")]
#[allow(unused_variables)]
pub fn model_selection_workloads(mini_batch: String, n: i32, k: i32, config_file: String) -> String {
    let mut task_map = HashMap::new();
    task_map.insert("mini_batch", mini_batch);
    task_map.insert("n", n.to_string());
    task_map.insert("k", k.to_string());
    task_map.insert("config_file", config_file);
    let task_json = json!(task_map).to_string();
    crate::bindings::ms::model_selection_workloads(&task_json).to_string()
}


// this two are filtering + refinement in GPU server
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "model_selection_trails")]
#[allow(unused_variables)]
pub fn model_selection_trails(mini_batch: String, time_budget: String, config_file: String) -> String {
    let mut task_map = HashMap::new();
    task_map.insert("mini_batch", mini_batch);
    task_map.insert("budget", time_budget);
    task_map.insert("config_file", config_file);
    let task_json = json!(task_map).to_string();
    crate::bindings::ms::model_selection_trails(&task_json).to_string()
}


#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "model_selection_trails_workloads")]
#[allow(unused_variables)]
pub fn model_selection_trails_workloads(mini_batch: String, n: i32, k: i32, config_file: String) -> String {
    let mut task_map = HashMap::new();
    task_map.insert("mini_batch", mini_batch);
    task_map.insert("n", n.to_string());
    task_map.insert("k", k.to_string());
    task_map.insert("config_file", config_file);
    let task_json = json!(task_map).to_string();
    crate::bindings::ms::model_selection_trails_workloads(&task_json).to_string()
}

// micro benchmarks
#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "benchmark_filtering_phase_latency")]
#[allow(unused_variables)]
pub fn benchmark_filtering_phase_latency(explore_models: i32, config_file: String) -> String {
    let mut task_map = HashMap::new();
    task_map.insert("explore_models", explore_models.to_string());
    task_map.insert("config_file", config_file);
    let task_json = json!(task_map).to_string();
    crate::bindings::ms::benchmark_filtering_phase_latency(&task_json).to_string()
}

#[cfg(feature = "python")]
#[pg_extern(immutable, parallel_safe, name = "benchmark_filtering_latency_in_db")]
#[allow(unused_variables)]
pub fn benchmark_filtering_latency_in_db(
    explore_models: i32, dataset: String, config_file: String) -> String {
    crate::bindings::ms::benchmark_filtering_latency_in_db(explore_models, &dataset, &config_file).to_string()
}





