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


# set PYTHONPATH
import sys

# Path you want to add
sys.path = [
    '/project/Trails/internal/ml/model_slicing',
    '/project/Trails/internal/ml/model_slicing/algorithm',
    '/project/Trails/internal/ml',
    '/usr/lib/python38.zip', '/usr/lib/python3.8', '/usr/lib/python3.8/lib-dynload', '/home/postgres/.local/lib/python3.8/site-packages', '/usr/local/lib/python3.8/dist-packages', '/usr/lib/python3/dist-packages']

import calendar
import os
import time
import json
import traceback
import orjson
from argparse import Namespace
from model_selection.shared_config import parse_config_arguments
from multiprocessing import shared_memory
import torch
from typing import Any, List, Dict, Tuple


def read_json(file_name):
    print(f"Loading {file_name}...")
    is_exist = os.path.exists(file_name)
    if is_exist:
        with open(file_name, 'r') as readfile:
            data = json.load(readfile)
        return data
    else:
        print(f"{file_name} is not exist")
        return {}


def exception_catcher(func):
    def wrapper(encoded_str: str):
        try:
            # each functon accepts a json string
            params = json.loads(encoded_str)
            config_file = params.get("config_file")

            # Parse the config file
            args = parse_config_arguments(config_file)

            # Set the environment variables
            ts = calendar.timegm(time.gmtime())
            os.environ.setdefault("base_dir", args.base_dir)
            os.environ.setdefault("log_logger_folder_name", args.log_folder)
            os.environ.setdefault("log_file_name", args.log_name + "_" + str(ts) + ".log")

            # Call the original function with the parsed parameters
            return func(params, args)
        except Exception as e:
            return orjson.dumps(
                {"Errored": traceback.format_exc()}).decode('utf-8')

    return wrapper


# Micro benchmarking filterting phaes
model = None
sliced_model = None
col_cardinalities = None
time_usage_dic = {}


@exception_catcher
def model_inference_load_model(params: dict, args: Namespace):
    global model, sliced_model, col_cardinalities
    from model_selection.src.logger import logger
    try:
        logger.info(f"Received parameters: {params}")

        from model_slicing.algorithm.src.data_loader import sql_attached_dataloader
        from model_slicing.algorithm.profile_model_clean import load_model
        # read saved col_cardinatlites file
        if col_cardinalities is None:
            col_cardinalities = read_json(params["col_cardinalities_file"])

        # read the model path,
        model_path = params["model_path"]

        # get the where condition
        where_cond = json.loads(params["where_cond"])
        # generate default sql and selected sql
        target_sql = [col[-1] for col in col_cardinalities]
        for col_index, value in where_cond.items():
            target_sql[int(col_index)] = value
        logger.info(f"target_sql encoding is: {target_sql}")

        if model is None:
            logger.info("Load model .....")
            model, config = load_model(model_path)
            model.eval()
            sliced_model = model.tailor_by_sql(torch.tensor(target_sql).reshape(1, -1))
            sliced_model.eval()
            logger.info("Load model Done!")
        else:
            logger.info("Skip Load model")
    except:
        logger.info(orjson.dumps(
            {"Errored": traceback.format_exc()}).decode('utf-8'))
    return orjson.dumps({"ok": 1}).decode('utf-8')


@exception_catcher
def model_inference_compute(params: dict, args: Namespace):
    global model, sliced_model, col_cardinalities, time_usage_dic
    from model_selection.src.logger import logger
    try:

        overall_begin = time.time()
        mini_batch = json.loads(params["mini_batch"])
        logger.info("-----" * 10)

        time_usage_dic = {}

        # logger.info(f"Received status: {mini_batch['status']}")
        # if mini_batch["status"] != 'success':
        #     raise Exception

        # todo: for credit datasets, it has 23 fields
        mini_batch_used = [mini_batch[i:i + 23] for i in range(0, len(mini_batch), 23)]

        begin = time.time()
        # pre-processing mini_batch
        transformed_data = torch.LongTensor(mini_batch_used)
        time_usage_dic["py_conver_to_tensor"] = time.time() - begin

        logger.info(f"transformed data size: {transformed_data.size()}")

        begin = time.time()
        y = sliced_model(transformed_data, None)
        time_usage_dic["py_compute"] = time.time() - begin
        logger.info(f"Prediction Results = {y.tolist()[:2]}...")

        logger.info("-----" * 10)
        overall_end = time.time()
        time_usage_dic["py_overall_duration"] = overall_end - overall_begin
        time_usage_dic["py_diff"] = time_usage_dic["py_overall_duration"] - \
                                    (time_usage_dic["py_conver_to_tensor"] + time_usage_dic["py_compute"])

        logger.info(f"time usage of inference {len(transformed_data)} rows is {time_usage_dic}")
    except:
        logger.info(orjson.dumps(
            {"Errored": traceback.format_exc()}).decode('utf-8'))

    return orjson.dumps({"model_outputs": 1}).decode('utf-8')


@exception_catcher
def model_inference_compute_shared_memory(params: dict, args: Namespace):
    global model, sliced_model, col_cardinalities, time_usage_dic
    from model_selection.src.logger import logger
    try:
        mini_batch_shared = get_data_from_shared_memory()
        logger.info(f"mini_batch_shared: {mini_batch_shared[:100]}")

        overall_begin = time.time()
        mini_batch = json.loads(mini_batch_shared)
        logger.info("-----" * 10)

        time_usage_dic = {}

        logger.info(f"Received status: {mini_batch['status']}")
        if mini_batch["status"] != 'success':
            raise Exception

        begin = time.time()
        # pre-processing mini_batch
        transformed_data = torch.LongTensor([
            [int(item.split(':')[0]) for item in sublist[2:]]
            for sublist in mini_batch["data"]])
        time_usage_dic["py_conver_to_tensor"] = time.time() - begin

        logger.info(f"transformed data size: {len(transformed_data)}")

        begin = time.time()
        y = sliced_model(transformed_data, None)
        time_usage_dic["py_compute"] = time.time() - begin
        logger.info(f"Prediction Results = {y.tolist()[:2]}...")

        logger.info("-----" * 10)
        overall_end = time.time()
        time_usage_dic["py_overall_duration"] = overall_end - overall_begin
        time_usage_dic["py_diff"] = time_usage_dic["py_overall_duration"] - \
                                    (time_usage_dic["py_conver_to_tensor"] + time_usage_dic["py_compute"])

        logger.info(f"time usage of inference {len(transformed_data)} rows is {time_usage_dic}")
    except:
        logger.info(orjson.dumps(
            {"Errored": traceback.format_exc()}).decode('utf-8'))

    return orjson.dumps({"model_outputs": 1}).decode('utf-8')


def decode_libsvm(columns):
    map_func = lambda pair: (int(pair[0]), float(pair[1]))
    # 0 is id, 1 is label
    id, value = zip(*map(lambda col: map_func(col.split(':')), columns[2:]))
    sample = {'id': list(id)}
    return sample


def pre_processing(mini_batch_data: List[Tuple]):
    """
    mini_batch_data: [('0', '0', '123:123', '123:123', '123:123',)
    """
    sample_lines = len(mini_batch_data)
    feat_id = []
    feat_value = []
    y = []
    for i in range(sample_lines):
        row_value = mini_batch_data[i]
        sample = decode_libsvm(row_value)
        feat_id.append(sample['id'])
    feat_id = torch.LongTensor(feat_id)
    return {'id': feat_id}


@exception_catcher
def model_inference_compute_shared_memory_write_once(params: dict, args: Namespace):
    global model, sliced_model, col_cardinalities, time_usage_dic
    from model_selection.src.logger import logger
    try:
        mini_batch_shared = get_data_from_shared_memory()
        logger.info(f"mini_batch_shared: <-{mini_batch_shared[:50]}->, type: {type(mini_batch_shared)}")

        overall_begin = time.time()
        mini_batch = json.loads(mini_batch_shared)
        logger.info("-----" * 10)

        time_usage_dic = {}

        begin = time.time()
        # pre-processing mini_batch
        transformed_data = pre_processing(mini_batch)['id']
        time_usage_dic["py_conver_to_tensor"] = time.time() - begin
        logger.info(f"transformed data size: {len(transformed_data)}")

        begin = time.time()
        y = sliced_model(transformed_data, None)
        time_usage_dic["py_compute"] = time.time() - begin
        logger.info(f"Prediction Results = {y.tolist()[:2]}...")

        logger.info("-----" * 10)
        overall_end = time.time()
        time_usage_dic["py_overall_duration"] = overall_end - overall_begin
        time_usage_dic["py_diff"] = time_usage_dic["py_overall_duration"] - \
                                    (time_usage_dic["py_conver_to_tensor"] + time_usage_dic["py_compute"])

        logger.info(f"time usage of inference {len(transformed_data)} rows is {time_usage_dic}")
    except:
        logger.info(orjson.dumps(
            {"Errored": traceback.format_exc()}).decode('utf-8'))

    return orjson.dumps({"model_outputs": 1}).decode('utf-8')

@exception_catcher
def model_inference_compute_shared_memory_write_once_int(params: dict, args: Namespace):
    global model, sliced_model, col_cardinalities, time_usage_dic
    from model_selection.src.logger import logger
    time_usage_dic = {}

    try:
        mini_batch_shared = get_data_from_shared_memory_int(int(params["rows"]))
        # logger.info(f"mini_batch_shared: <-{mini_batch_shared[:50]}->, type: {type(mini_batch_shared)}")
        logger.info(f"mini_batch_shared: <-{mini_batch_shared}->, type: {type(mini_batch_shared)}")

        overall_begin = time.time()
        logger.info("-----" * 10)

        begin = time.time()
        # pre-processing mini_batch
        transformed_data = torch.LongTensor(mini_batch_shared)
        time_usage_dic["py_conver_to_tensor"] = time.time() - begin
        logger.info(f"transformed data size: {transformed_data.size()}")

        begin = time.time()
        y = sliced_model(transformed_data, None)
        time_usage_dic["py_compute"] = time.time() - begin
        logger.info(f"Prediction Results = {y.tolist()[:2]}...")

        logger.info("-----" * 10)
        overall_end = time.time()
        time_usage_dic["py_overall_duration"] = overall_end - overall_begin
        time_usage_dic["py_diff"] = time_usage_dic["py_overall_duration"] - \
                                    (time_usage_dic["py_conver_to_tensor"] + time_usage_dic["py_compute"])

        logger.info(f"time usage of inference {len(transformed_data)} rows is {time_usage_dic}")
    except:
        logger.info(orjson.dumps(
            {"Errored": traceback.format_exc()}).decode('utf-8'))

    return orjson.dumps({"model_outputs": 1}).decode('utf-8')


def records_results(params: str):
    global time_usage_dic
    from model_selection.src.logger import logger
    try:
        params = json.loads(params)
        params.update(time_usage_dic)
        logger.info(f"final result = {params}")
    except:
        logger.info(orjson.dumps(
            {"Errored": traceback.format_exc()}).decode('utf-8'))
    return orjson.dumps({"Done": 1}).decode('utf-8')


def get_data_from_shared_memory(shmem_name="my_shmem"):
    # Open existing shared memory segment
    shm = shared_memory.SharedMemory(name="my_shared_memory")
    # Read data
    data = shm.buf.tobytes().decode()
    # Close
    shm.close()
    return data.rstrip('\x00')


import numpy as np

def get_data_from_shared_memory_int(n_rows):
    # Connect to existing shared memory by name
    shm = shared_memory.SharedMemory(name="my_shared_memory")
    # Map the shared memory to a numpy array. Assuming i32 integers.
    data = np.frombuffer(shm.buf, dtype=np.int32)
    # Reshape the 1D array to have n_rows and let numpy infer the number of columns
    data = data.reshape(n_rows, -1)
    return data

