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

use serde_json::json;
use std::collections::HashMap;
use pgrx::prelude::*;
use crate::bindings::ml_register::PY_MODULE;
use crate::bindings::ml_register::run_python_function;
use std::time::{Instant};
use shared_memory::*;

pub fn profiling_filtering_phase(
    task: &String
) -> serde_json::Value {
    run_python_function(&PY_MODULE, task, "profiling_filtering_phase")
}


pub fn profiling_refinement_phase(
    task: &String
) -> serde_json::Value {
    run_python_function(&PY_MODULE, task, "profiling_refinement_phase")
}


pub fn coordinator(
    task: &String
) -> serde_json::Value {
    run_python_function(&PY_MODULE, task, "coordinator")
}


pub fn filtering_phase(
    task: &String
) -> serde_json::Value {
    run_python_function(&PY_MODULE, task, "filtering_phase")
}


pub fn refinement_phase() -> serde_json::Value {
    let task = "refinement_phase".to_string();
    run_python_function(&PY_MODULE, &task, "refinement_phase")
}


// this two are filtering + refinement in UDF runtime
pub fn model_selection(
    task: &String
) -> serde_json::Value {
    run_python_function(&PY_MODULE, task, "model_selection")
}


pub fn model_selection_workloads(
    task: &String
) -> serde_json::Value {
    run_python_function(&PY_MODULE, task, "model_selection_workloads")
}


// this two are filtering + refinement in GPU server
pub fn model_selection_trails(
    task: &String
) -> serde_json::Value {
    run_python_function(&PY_MODULE, task, "model_selection_trails")
}


pub fn model_selection_trails_workloads(
    task: &String
) -> serde_json::Value {
    run_python_function(&PY_MODULE, task, "model_selection_trails_workloads")
}

// micro benchmarks
// this is query data in filtering phase via sql
pub fn benchmark_filtering_phase_latency(
    task: &String
) -> serde_json::Value {
    run_python_function(&PY_MODULE, task, "benchmark_filtering_phase_latency")
}

// this is query data in filtering phase via spi
pub fn benchmark_filtering_latency_in_db(
    explore_models: i32, dataset: &String, batch_size_m: i32, config_file: &String) -> serde_json::Value {
    let mut return_result = HashMap::new();


    let mut total_columns: i32 = 0;
    match dataset.as_str() {  // assuming dataset is a String
        "frappe" => total_columns = 12,
        "criteo" => total_columns = 41,
        "uci_diabetes" => total_columns = 45,
        _ => {}
    }

    let mut num_columns: i64 = 0;
    match dataset.as_str() {  // assuming dataset is a String
        "frappe" => num_columns = 10 * 2 + 1,
        "criteo" => num_columns = 39 * 2 + 1,
        "uci_diabetes" => num_columns = 43 * 2 + 1,
        _ => {}
    }

    let batch_size: i64 = batch_size_m as i64;

    let call_time_begin = Instant::now();
    for _ in 1..=5000 {
        run_python_function(
            &PY_MODULE,
            &"".to_string(),
            "measure_call_overheads");
    }
    let _end_time = Instant::now();
    let call_time = _end_time.duration_since(call_time_begin).as_secs_f64();
    return_result.insert("call_time", call_time.to_string());


    let overall_start_time = Instant::now();

    let mut last_id = 0;
    let mut eva_results = serde_json::Value::Null; // Initializing the eva_results

    // Step 3: Putting all data to he shared memory
    let shmem_name = "my_shared_memory";
    let my_shmem = ShmemConf::new()
        .size((4 * batch_size * num_columns) as usize)
        .os_id(shmem_name)
        .create()
        .unwrap();

    let mut numbers: Vec<f32> = Vec::with_capacity((num_columns - 1) as usize );

    let _ = Spi::connect(|client| {
        for i in 1..explore_models + 1 {
            // Step 1: Initialize State in Python
            let mut task_map = HashMap::new();
            task_map.insert("config_file", config_file.clone());
            task_map.insert("dataset", dataset.clone());
            task_map.insert("eva_results", eva_results.to_string());
            let task_json = json!(task_map).to_string();

            // here it cache a state
            let sample_result = run_python_function(
                &PY_MODULE,
                &task_json,
                "in_db_filtering_state_init");

            // 2. query data via SPI
            let start_time = Instant::now();
            let mut mini_batch = Vec::new();

            let query = format!("SELECT * FROM {}_train WHERE id > {} ORDER BY id ASC LIMIT {}", dataset, last_id, batch_size);
            let mut cursor = client.open_cursor(&query, None);
            let table = match cursor.fetch(batch_size) {
                Ok(table) => table,
                Err(e) => return Err(e.to_string()), // Convert the error to a string and return
            };

            for row in table.into_iter() {
                // add primary key
                let val = row.get::<i32>(1)
                    .expect("Failed to retrieve value")  // This will panic if it encounters `Err`
                    .expect("Retrieved value is NULL");  // This will panic if it encounters `None`

                if val > 80000 {
                    last_id = 0;
                } else {
                    last_id = val;
                }

                // add label
                if let Ok(Some(col1)) = row.get::<i32>(2) {
                    mini_batch.push(col1 as f32);
                };

                numbers.clear();
                for i in 3..= total_columns as usize {
                    if let Some(s) = row.get::<&str>(i).ok().flatten() { // Ensuring it's Some(&str)
                        for part in s.split(':') {
                            match part.parse::<f32>() {
                                Ok(num) => numbers.push(num),
                                Err(_) => eprintln!("Failed to parse part as f32"), // Handle the error as appropriate for your application.
                            }
                        }
                    }
                }

                mini_batch.extend_from_slice(&numbers);
            }

            unsafe {
                let shmem_ptr = my_shmem.as_ptr() as *mut f32;
                // Copy data into shared memory
                std::ptr::copy_nonoverlapping(
                    mini_batch.as_ptr(),
                    shmem_ptr as *mut f32,
                    mini_batch.len(),
                );
            }

            let end_time = Instant::now();
            let elapsed_time = end_time.duration_since(start_time);
            let elapsed_seconds = elapsed_time.as_secs_f64();

            // Step 3: model evaluate in Python
            let mut eva_task_map = HashMap::new();
            eva_task_map.insert("config_file", config_file.clone());
            eva_task_map.insert("sample_result", sample_result.to_string());
            eva_task_map.insert("spi_seconds", elapsed_seconds.to_string());
            eva_task_map.insert("rows", batch_size.to_string());
            eva_task_map.insert("model_index", i.to_string());
            let eva_task_json = json!(eva_task_map).to_string(); // Corrected this line

            eva_results = run_python_function(
                &PY_MODULE,
                &eva_task_json,
                "in_db_filtering_evaluate");

            // debug the fetched data
            // if i == 1{
            //     let serialized_data = json!(mini_batch).to_string();
            //     return_result.insert("serialized_data", serialized_data);
            // };
        };
        Ok(())
    });

    let overall_end_time = Instant::now();
    let overall_elapsed_time = overall_end_time.duration_since(overall_start_time);
    let overall_elapsed_seconds = overall_elapsed_time.as_secs_f64();

    return_result.insert("overall time usage", overall_elapsed_seconds.to_string());

    let mut record_task_map = HashMap::new();
    record_task_map.insert("config_file", config_file.clone());
    record_task_map.insert("dataset", dataset.clone());
    let record_task_json = json!(record_task_map).to_string();
    run_python_function(
        &PY_MODULE,
        &record_task_json,
        "records_results");

    // Step 4: Return to PostgresSQL
    return serde_json::json!(return_result);
}

