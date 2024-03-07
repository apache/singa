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
use std::ffi::c_long;
use pgrx::prelude::*;
use crate::bindings::ml_register::PY_MODULE_INFERENCE;
use crate::bindings::ml_register::run_python_function;
use std::time::{Instant};
use shared_memory::*;


pub fn run_inference_shared_memory(
    dataset: &String,
    condition: &String,
    config_file: &String,
    col_cardinalities_file: &String,
    model_path: &String,
    sql: &String,
    batch_size: i32,
) -> serde_json::Value {
    let mut response = HashMap::new();

    let overall_start_time = Instant::now();

    let mut last_id = 0;

    // Step 1: load model and columns etc
    let mut task_map = HashMap::new();
    task_map.insert("where_cond", condition.clone());
    task_map.insert("config_file", config_file.clone());
    task_map.insert("col_cardinalities_file", col_cardinalities_file.clone());
    task_map.insert("model_path", model_path.clone());
    let task_json = json!(task_map).to_string();
    // here it cache a state
    run_python_function(
        &PY_MODULE_INFERENCE,
        &task_json,
        "model_inference_load_model");

    let _end_time = Instant::now();
    let model_init_time = _end_time.duration_since(overall_start_time).as_secs_f64();
    response.insert("model_init_time", model_init_time.clone());

    // Step 2: query data via SPI
    let start_time = Instant::now();
    let results: Result<Vec<Vec<String>>, String> = Spi::connect(|client| {
        let query = format!("SELECT * FROM {}_train {} LIMIT {}",
                            dataset, sql, batch_size);
        let mut cursor = client.open_cursor(&query, None);
        let table = match cursor.fetch(batch_size as c_long) {
            Ok(table) => table,
            Err(e) => return Err(e.to_string()), // Convert the error to a string and return
        };

        let mut mini_batch = Vec::new();

        for row in table.into_iter() {
            let mut each_row = Vec::new();
            // add primary key
            let col0 = match row.get::<i32>(1) {
                Ok(Some(val)) => {
                    // Update last_id with the retrieved value
                    if val > 100000 {
                        last_id = 0;
                    } else {
                        last_id = val
                    }
                    val.to_string()
                }
                Ok(None) => "".to_string(), // Handle the case when there's no valid value
                Err(e) => e.to_string(),
            };
            each_row.push(col0);
            // add label
            let col1 = match row.get::<i32>(2) {
                Ok(val) => val.map(|i| i.to_string()).unwrap_or_default(),
                Err(e) => e.to_string(),
            };
            each_row.push(col1);
            // add fields
            let texts: Vec<String> = (3..row.columns() + 1)
                .filter_map(|i| {
                    match row.get::<&str>(i) {
                        Ok(Some(s)) => Some(s.to_string()),
                        Ok(None) => None,
                        Err(e) => Some(e.to_string()),  // Convert error to string
                    }
                }).collect();
            each_row.extend(texts);
            mini_batch.push(each_row)
        }
        // return
        Ok(mini_batch)
    });
    // serialize the mini-batch data
    let tup_table = match results {
        Ok(data) => {
            serde_json::json!({
                        "status": "success",
                        "data": data
                    })
        }
        Err(e) => {
            serde_json::json!({
                    "status": "error",
                    "message": format!("Error while connecting: {}", e)
                })
        }
    };
    let mini_batch_json = tup_table.to_string();

    let end_time = Instant::now();
    let data_query_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("data_query_time", data_query_time.clone());


    let start_time = Instant::now();
    // Set an identifier for the shared memory
    let shmem_name = "my_shared_memory";
    let my_shmem = ShmemConf::new()
        .size(tup_table.to_string().len())
        .os_id(shmem_name)
        .create()
        .unwrap();

    // Use unsafe to access and write to the raw memory
    let data_to_write = mini_batch_json.as_bytes();
    unsafe {
        // Get the raw pointer to the shared memory
        let shmem_ptr = my_shmem.as_ptr() as *mut u8;
        // Copy data into the shared memory
        std::ptr::copy_nonoverlapping(
            data_to_write.as_ptr(), shmem_ptr, data_to_write.len());
    }

    let end_time = Instant::now();
    let data_copy_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("data_copy", data_copy_time.clone());

    let start_time = Instant::now();
    // Step 3: model evaluate in Python
    let mut eva_task_map = HashMap::new();
    eva_task_map.insert("config_file", config_file.clone());
    eva_task_map.insert("spi_seconds", data_query_time.to_string());

    let eva_task_json = json!(eva_task_map).to_string(); // Corrected this line

    run_python_function(
        &PY_MODULE_INFERENCE,
        &eva_task_json,
        "model_inference_compute_shared_memory");

    let end_time = Instant::now();
    let python_compute_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("python_compute_time", python_compute_time.clone());

    let overall_end_time = Instant::now();
    let overall_elapsed_time = overall_end_time.duration_since(overall_start_time).as_secs_f64();
    let diff_time = model_init_time + data_query_time + data_copy_time + python_compute_time - overall_elapsed_time;

    response.insert("overall_query_latency", overall_elapsed_time.clone());
    response.insert("diff", diff_time.clone());

    // Step 4: Return to PostgresSQL
    return serde_json::json!(response);
}


pub fn run_inference(
    dataset: &String,
    condition: &String,
    config_file: &String,
    col_cardinalities_file: &String,
    model_path: &String,
    sql: &String,
    batch_size: i32,
) -> serde_json::Value {
    let mut response = HashMap::new();

    let overall_start_time = Instant::now();

//     let mut last_id = 0;

    // Step 1: load model and columns etc
    let mut task_map = HashMap::new();
    task_map.insert("where_cond", condition.clone());
    task_map.insert("config_file", config_file.clone());
    task_map.insert("col_cardinalities_file", col_cardinalities_file.clone());
    task_map.insert("model_path", model_path.clone());
    let task_json = json!(task_map).to_string();
    // here it cache a state
    run_python_function(
        &PY_MODULE_INFERENCE,
        &task_json,
        "model_inference_load_model");

    let _end_time = Instant::now();
    let model_init_time = _end_time.duration_since(overall_start_time).as_secs_f64();
    response.insert("model_init_time", model_init_time.clone());

    // Step 2: query data via SPI
    let start_time = Instant::now();
    let mut all_rows = Vec::new();
    let _ = Spi::connect(|client| {
        let query = format!("SELECT * FROM {}_int_train {} LIMIT {}", dataset, sql, batch_size);
        let mut cursor = client.open_cursor(&query, None);
        let table = match cursor.fetch(batch_size as c_long) {
            Ok(table) => table,
            Err(e) => return Err(e.to_string()),
        };

        let end_time = Instant::now();
        let data_query_time_spi = end_time.duration_since(start_time).as_secs_f64();
        response.insert("data_query_time_spi", data_query_time_spi);

        // todo: nl: this part can must be optimized, since i go through all of those staff.
        for row in table.into_iter() {
            for i in 3..=row.columns() {
                match row.get::<i32>(i) {
                    Ok(Some(val)) => all_rows.push(val), // Handle the case when a valid i32 is obtained
                    Ok(None) => {
                        // Handle the case when the value is missing or erroneous
                        // For example, you can add a default value, like -1
                        all_rows.push(-1);
                    }
                    Err(e) => {
                        // Handle the error, e.g., log it or handle it in some way
                        eprintln!("Error fetching value: {:?}", e);
                    }
                }
            }
        }
        // Return OK or some status
        Ok(())
    });

    let mini_batch_json = serde_json::to_string(&all_rows).unwrap();

    let end_time = Instant::now();
    let data_query_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("data_query_time", data_query_time.clone());

    let start_time = Instant::now();
    // Step 3: model evaluate in Python
    let mut eva_task_map = HashMap::new();
    eva_task_map.insert("config_file", config_file.clone());
    eva_task_map.insert("mini_batch", mini_batch_json);
    eva_task_map.insert("spi_seconds", data_query_time.to_string());

    let eva_task_json = json!(eva_task_map).to_string(); // Corrected this line

    run_python_function(
        &PY_MODULE_INFERENCE,
        &eva_task_json,
        "model_inference_compute");

    let end_time = Instant::now();
    let python_compute_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("python_compute_time", python_compute_time.clone());

    let overall_end_time = Instant::now();
    let overall_elapsed_time = overall_end_time.duration_since(overall_start_time).as_secs_f64();
    let diff_time = model_init_time + data_query_time + python_compute_time - overall_elapsed_time;

    response.insert("overall_query_latency", overall_elapsed_time.clone());
    response.insert("diff", diff_time.clone());

    let response_json = json!(response).to_string();
    run_python_function(
        &PY_MODULE_INFERENCE,
        &response_json,
        "records_results");

    // Step 4: Return to PostgresSQL
    return serde_json::json!(response);
}


pub fn run_inference_shared_memory_write_once(
    dataset: &String,
    condition: &String,
    config_file: &String,
    col_cardinalities_file: &String,
    model_path: &String,
    sql: &String,
    batch_size: i32,
) -> serde_json::Value {
    let mut response = HashMap::new();

    let overall_start_time = Instant::now();

    let mut last_id = 0;

    // Step 1: load model and columns etc
    let mut task_map = HashMap::new();
    task_map.insert("where_cond", condition.clone());
    task_map.insert("config_file", config_file.clone());
    task_map.insert("col_cardinalities_file", col_cardinalities_file.clone());
    task_map.insert("model_path", model_path.clone());
    let task_json = json!(task_map).to_string();
    // here it cache a state
    run_python_function(
        &PY_MODULE_INFERENCE,
        &task_json,
        "model_inference_load_model");

    let _end_time = Instant::now();
    let model_init_time = _end_time.duration_since(overall_start_time).as_secs_f64();
    response.insert("model_init_time", model_init_time.clone());

    // Step 2: query data via SPI
    let start_time = Instant::now();
    // Allocate shared memory in advance
    // Set an identifier for the shared memory
    let shmem_name = "my_shared_memory";

    // Pre-allocate a size for shared memory (this might need some logic to determine a reasonable size)
    let avg_row_size = 120;
    let shmem_size = (1.5 * (avg_row_size * batch_size as usize) as f64) as usize;
    let my_shmem = ShmemConf::new()
        .size(shmem_size)
        .os_id(shmem_name)
        .create()
        .unwrap();

    let shmem_ptr = my_shmem.as_ptr() as *mut u8;

    let end_time = Instant::now();
    let mem_allocate_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("mem_allocate_time", mem_allocate_time.clone());

    let start_time = Instant::now();
    // Use unsafe to access and write to the raw memory
    unsafe {
        let _ = Spi::connect(|client| {
            let query = format!("SELECT * FROM {}_train {} LIMIT {}", dataset, sql, batch_size);
            let mut cursor = client.open_cursor(&query, None);
            let table = match cursor.fetch(batch_size as c_long) {
                Ok(table) => table,
                Err(e) => return Err(e.to_string()),
            };

            let end_time = Instant::now();
            let data_query_time_spi = end_time.duration_since(start_time).as_secs_f64();
            response.insert("data_query_time_spi", data_query_time_spi.clone());

            let mut offset = 0;  // Keep track of how much we've written to shared memory

            // Write the opening square bracket
            shmem_ptr.offset(offset as isize).write(b"["[0]);
            offset += 1;

            let mut is_first_row = true;
            for row in table.into_iter() {

                // If not the first row, write a comma before the next row's data
                if !is_first_row {
                    shmem_ptr.offset(offset as isize).write(b","[0]);
                    offset += 1;
                } else {
                    is_first_row = false;
                }

                let mut each_row = Vec::new();
                // add primary key
                let col0 = match row.get::<i32>(1) {
                    Ok(Some(val)) => {
                        // Update last_id with the retrieved value
                        if val > 100000 {
                            last_id = 0;
                        } else {
                            last_id = val
                        }
                        val.to_string()
                    }
                    Ok(None) => "".to_string(), // Handle the case when there's no valid value
                    Err(e) => e.to_string(),
                };
                each_row.push(col0);
                // add label
                let col1 = match row.get::<i32>(2) {
                    Ok(val) => val.map(|i| i.to_string()).unwrap_or_default(),
                    Err(e) => e.to_string(),
                };
                each_row.push(col1);
                // add fields
                let texts: Vec<String> = (3..row.columns() + 1)
                    .filter_map(|i| {
                        match row.get::<&str>(i) {
                            Ok(Some(s)) => Some(s.to_string()),
                            Ok(None) => None,
                            Err(e) => Some(e.to_string()),  // Convert error to string
                        }
                    }).collect();
                each_row.extend(texts);

                // Serialize each row into shared memory
                let serialized_row = serde_json::to_string(&each_row).unwrap();
                let bytes = serialized_row.as_bytes();

                // Check if there's enough space left in shared memory
                if offset + bytes.len() > shmem_size {
                    // Handle error: not enough space in shared memory
                    return Err("Shared memory exceeded estimated size.".to_string());
                }

                // Copy the serialized row into shared memory
                std::ptr::copy_nonoverlapping(bytes.as_ptr(),
                                              shmem_ptr.offset(offset as isize),
                                              bytes.len());
                offset += bytes.len();
            }
            // Write the closing square bracket after all rows
            shmem_ptr.offset(offset as isize).write(b"]"[0]);

            // Return OK or some status
            Ok(())
        });
    }

    let end_time = Instant::now();
    let data_query_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("data_query_time", data_query_time.clone());

    let start_time = Instant::now();
    // Step 3: model evaluate in Python
    let mut eva_task_map = HashMap::new();
    eva_task_map.insert("config_file", config_file.clone());
    eva_task_map.insert("spi_seconds", data_query_time.to_string());

    let eva_task_json = json!(eva_task_map).to_string(); // Corrected this line

    run_python_function(
        &PY_MODULE_INFERENCE,
        &eva_task_json,
        "model_inference_compute_shared_memory_write_once");

    let end_time = Instant::now();
    let python_compute_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("python_compute_time", python_compute_time.clone());

    let overall_end_time = Instant::now();
    let overall_elapsed_time = overall_end_time.duration_since(overall_start_time).as_secs_f64();
    let diff_time = model_init_time + data_query_time + python_compute_time - overall_elapsed_time;

    response.insert("overall_query_latency", overall_elapsed_time.clone());
    response.insert("diff", diff_time.clone());


    let response_json = json!(response).to_string();
    run_python_function(
        &PY_MODULE_INFERENCE,
        &response_json,
        "records_results");

    // Step 4: Return to PostgresSQL
    return serde_json::json!(response);
}


pub fn run_inference_shared_memory_write_once_int_exp(
    dataset: &String,
    condition: &String,
    config_file: &String,
    col_cardinalities_file: &String,
    model_path: &String,
    sql: &String,
    batch_size: i32,
) -> serde_json::Value {
    let mut response = HashMap::new();
    // let mut response_log = HashMap::new();

    let mut num_columns: i32 = 0;
    match dataset.as_str() {  // assuming dataset is a String
        "frappe" => num_columns = 12,
        "adult" => num_columns = 15,
        "cvd" => num_columns = 13,
        "bank" => num_columns = 18,
        "census" => num_columns = 41+2,
        "credit" => num_columns = 23+2,
        "diabetes" => num_columns = 48+2,
        "hcdr" => num_columns = 69+2,
        _ => {},
    }

    let overall_start_time = Instant::now();

    // Step 1: load model and columns etc
    let mut task_map = HashMap::new();
    task_map.insert("where_cond", condition.clone());
    task_map.insert("config_file", config_file.clone());
    task_map.insert("col_cardinalities_file", col_cardinalities_file.clone());
    task_map.insert("model_path", model_path.clone());
    let task_json = json!(task_map).to_string();
    // here it cache a state
    run_python_function(
        &PY_MODULE_INFERENCE,
        &task_json,
        "model_inference_load_model");

    let _end_time = Instant::now();
    let model_init_time = _end_time.duration_since(overall_start_time).as_secs_f64();
    response.insert("model_init_time", model_init_time.clone());


    // Step 1: query data
    let start_time = Instant::now();
    let mut all_rows = Vec::new();

    let _ = Spi::connect(|client| {
        let query = format!("SELECT * FROM {}_int_train {} LIMIT {}", dataset, sql, batch_size);
        let mut cursor = client.open_cursor(&query, None);
        let table = match cursor.fetch(batch_size as c_long) {
            Ok(table) => table,
            Err(e) => return Err(e.to_string()),
        };

        let end_time = Instant::now();
        let data_query_time_spi = end_time.duration_since(start_time).as_secs_f64();
        response.insert("data_query_time_spi", data_query_time_spi);

        let mut t1: f64 = 0.0;
        // todo: nl: this part can must be optimized, since i go through all of those staff.
        let start_time_3 = Instant::now();
        for row in table.into_iter() {
            for i in 3..=num_columns as usize {
                let start_time_min = Instant::now();
                if let Ok(Some(val)) = row.get::<i32>(i) {
                    all_rows.push(val);
                }
                let end_time_min = Instant::now();
                let data_query_time_min = end_time_min.duration_since(start_time_min).as_secs_f64();
                t1 += data_query_time_min;
            }
        }
        let end_time_min3 = Instant::now();
        let data_query_time_min3 = end_time_min3.duration_since(start_time_3).as_secs_f64();

        response.insert("data_query_time3", data_query_time_min3.clone());
        response.insert("data_query_time2", t1.clone());

        // Return OK or some status
        Ok(())
    });
    let end_time = Instant::now();
    let data_query_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("data_query_time", data_query_time.clone());


    // log the query datas
    // let serialized_row = serde_json::to_string(&all_rows).unwrap();
    // response_log.insert("query_data", serialized_row);

    // Step 3: Putting all data to he shared memory
    let start_time = Instant::now();
    let shmem_name = "my_shared_memory";
    let my_shmem = ShmemConf::new()
        .size(4 * all_rows.len())
        .os_id(shmem_name)
        .create()
        .unwrap();
    let shmem_ptr = my_shmem.as_ptr() as *mut i32;

    unsafe {
        // Copy data into shared memory
        std::ptr::copy_nonoverlapping(
            all_rows.as_ptr(),
            shmem_ptr as *mut i32,
            all_rows.len(),
        );
    }
    let end_time = Instant::now();
    let mem_allocate_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("mem_allocate_time", mem_allocate_time.clone());


    let start_time = Instant::now();
    // Step 3: model evaluate in Python
    let mut eva_task_map = HashMap::new();
    eva_task_map.insert("config_file", config_file.clone());
    eva_task_map.insert("spi_seconds", data_query_time.to_string());
    eva_task_map.insert("rows", batch_size.to_string());

    let eva_task_json = json!(eva_task_map).to_string(); // Corrected this line

    run_python_function(
        &PY_MODULE_INFERENCE,
        &eva_task_json,
        "model_inference_compute_shared_memory_write_once_int");

    let end_time = Instant::now();
    let python_compute_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("python_compute_time", python_compute_time.clone());

    let overall_end_time = Instant::now();
    let overall_elapsed_time = overall_end_time.duration_since(overall_start_time).as_secs_f64();
    let diff_time = model_init_time + data_query_time + python_compute_time - overall_elapsed_time;

    response.insert("overall_query_latency", overall_elapsed_time.clone());
    response.insert("diff", diff_time.clone());

    let response_json = json!(response).to_string();
    run_python_function(
        &PY_MODULE_INFERENCE,
        &response_json,
        "records_results");

    // Step 4: Return to PostgresSQL
    return serde_json::json!(response);
}

pub fn run_inference_shared_memory_write_once_int(
    dataset: &String,
    condition: &String,
    config_file: &String,
    col_cardinalities_file: &String,
    model_path: &String,
    sql: &String,
    batch_size: i32,
) -> serde_json::Value {
    let mut response = HashMap::new();

    let mut num_columns: i32 = 0;
    match dataset.as_str() {  // assuming dataset is a String
        "frappe" => num_columns = 12,
        "adult" => num_columns = 15,
        "cvd" => num_columns = 13,
        "bank" => num_columns = 18,
        "census" => num_columns = 41+2,
        "credit" => num_columns = 23+2,
        "diabetes" => num_columns = 48+2,
        "hcdr" => num_columns = 69+2,
        _ => {},
    }

    let overall_start_time = Instant::now();

    // Step 1: load model and columns etc
    let mut task_map = HashMap::new();
    task_map.insert("where_cond", condition.clone());
    task_map.insert("config_file", config_file.clone());
    task_map.insert("col_cardinalities_file", col_cardinalities_file.clone());
    task_map.insert("model_path", model_path.clone());
    let task_json = json!(task_map).to_string();
    // here it cache a state
    run_python_function(
        &PY_MODULE_INFERENCE,
        &task_json,
        "model_inference_load_model");

    let _end_time = Instant::now();
    let model_init_time = _end_time.duration_since(overall_start_time).as_secs_f64();
    response.insert("model_init_time", model_init_time.clone());


    // Step 1: query data
    let start_time = Instant::now();
    let mut all_rows = Vec::new();
    let _ = Spi::connect(|client| {
        let query = format!("SELECT * FROM {}_int_train {} LIMIT {}", dataset, sql, batch_size);
        let mut cursor = client.open_cursor(&query, None);
        let table = match cursor.fetch(batch_size as c_long) {
            Ok(table) => table,
            Err(e) => return Err(e.to_string()),
        };
        let end_time = Instant::now();
        let data_query_time_spi = end_time.duration_since(start_time).as_secs_f64();
        response.insert("data_query_time_spi", data_query_time_spi);

        // todo: nl: this part can must be optimized, since i go through all of those staff.
        let start_time_3 = Instant::now();
        for row in table.into_iter() {
            for i in 3..= num_columns as usize {
                if let Ok(Some(val)) = row.get::<i32>(i) {
                    all_rows.push(val);
                }
            }
        }
        let end_time_min3 = Instant::now();
        let data_query_time_min3 = end_time_min3.duration_since(start_time_3).as_secs_f64();
        response.insert("data_type_convert_time", data_query_time_min3.clone());

        // Return OK or some status
        Ok(())
    });
    let end_time = Instant::now();
    let data_query_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("data_query_time", data_query_time.clone());


    // log the query datas
    // let serialized_row = serde_json::to_string(&all_rows).unwrap();
    // response_log.insert("query_data", serialized_row);

    // Step 3: Putting all data to he shared memory
    let start_time = Instant::now();
    let shmem_name = "my_shared_memory";
    let my_shmem = ShmemConf::new()
        .size(4 * all_rows.len())
        .os_id(shmem_name)
        .create()
        .unwrap();
    let shmem_ptr = my_shmem.as_ptr() as *mut i32;

    unsafe {
        // Copy data into shared memory
        std::ptr::copy_nonoverlapping(
            all_rows.as_ptr(),
            shmem_ptr as *mut i32,
            all_rows.len(),
        );
    }
    let end_time = Instant::now();
    let mem_allocate_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("mem_allocate_time", mem_allocate_time.clone());


    let start_time = Instant::now();
    // Step 3: model evaluate in Python
    let mut eva_task_map = HashMap::new();
    eva_task_map.insert("config_file", config_file.clone());
    eva_task_map.insert("spi_seconds", data_query_time.to_string());
    eva_task_map.insert("rows", batch_size.to_string());

    let eva_task_json = json!(eva_task_map).to_string(); // Corrected this line

    run_python_function(
        &PY_MODULE_INFERENCE,
        &eva_task_json,
        "model_inference_compute_shared_memory_write_once_int");

    let end_time = Instant::now();
    let python_compute_time = end_time.duration_since(start_time).as_secs_f64();
    response.insert("python_compute_time", python_compute_time.clone());

    let overall_end_time = Instant::now();
    let overall_elapsed_time = overall_end_time.duration_since(overall_start_time).as_secs_f64();
    let diff_time = model_init_time + data_query_time + python_compute_time - overall_elapsed_time;

    response.insert("overall_query_latency", overall_elapsed_time.clone());
    response.insert("diff", diff_time.clone());

    let response_json = json!(response).to_string();
    run_python_function(
        &PY_MODULE_INFERENCE,
        &response_json,
        "records_results");

    // Step 4: Return to PostgresSQL
    return serde_json::json!(response);
}


pub fn init_model(
    condition: &String,
    config_file: &String,
    col_cardinalities_file: &String,
    model_path: &String,
) -> serde_json::Value {
    let overall_start_time = Instant::now();
    // Step 1: load model and columns etc
    let mut task_map = HashMap::new();
    task_map.insert("where_cond", condition.clone());
    task_map.insert("config_file", config_file.clone());
    task_map.insert("col_cardinalities_file", col_cardinalities_file.clone());
    task_map.insert("model_path", model_path.clone());
    let task_json = json!(task_map).to_string();
    // here it cache a state
    run_python_function(
        &PY_MODULE_INFERENCE,
        &task_json,
        "model_inference_load_model");
    let _end_time = Instant::now();
    let model_init_time = _end_time.duration_since(overall_start_time).as_secs_f64();
    return serde_json::json!(model_init_time);
}