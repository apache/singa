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

import calendar
import os
import time
import requests
import json
from typing import List, Dict
import torch
from torch.utils.data import Dataset, DataLoader
import traceback
import orjson
from argparse import Namespace
from shared_config import parse_config_arguments


def exception_catcher(func):
    def wrapper(encoded_str: str):
        global_res = "NA, "
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
            global_res = func(params, args)
            return global_res
        except Exception as e:
            return orjson.dumps(
                {"res": global_res, "Errored": traceback.format_exc()}).decode('utf-8')

    return wrapper


class LibsvmDataset(Dataset):
    """ Dataset loader for Libsvm data format """

    @staticmethod
    def decode_libsvm(columns):
        map_func = lambda pair: (int(pair[0]), float(pair[1]))
        id, value = zip(*map(lambda col: map_func(col.split(':')), columns[:-1]))
        sample = {'id': torch.LongTensor(id),
                  'value': torch.FloatTensor(value),
                  'y': float(columns[-1])}
        return sample

    @staticmethod
    def pre_processing(mini_batch_data: List[Dict]):
        sample_lines = len(mini_batch_data)
        nfields = len(mini_batch_data[0].keys()) - 1
        feat_id = torch.LongTensor(sample_lines, nfields)
        feat_value = torch.FloatTensor(sample_lines, nfields)
        y = torch.FloatTensor(sample_lines)

        for i in range(sample_lines):
            row_value = mini_batch_data[i].values()
            sample = LibsvmDataset.decode_libsvm(list(row_value))
            feat_id[i] = sample['id']
            feat_value[i] = sample['value']
            y[i] = sample['y']
        return feat_id, feat_value, y, sample_lines

    def __init__(self, mini_batch_data: List[Dict]):
        self.feat_id, self.feat_value, self.y, self.nsamples = \
            LibsvmDataset.pre_processing(mini_batch_data)

    def __len__(self):
        return self.nsamples

    def __getitem__(self, idx):
        return {'id': self.feat_id[idx],
                'value': self.feat_value[idx],
                'y': self.y[idx]}


def generate_dataloader(mini_batch_data, args):
    from src.logger import logger
    logger.info(f"Begin to preprocessing dataset")
    begin_time = time.time()
    dataloader = DataLoader(LibsvmDataset(mini_batch_data),
                            batch_size=args.batch_size,
                            shuffle=True)
    logger.info(f"Preprocessing dataset Done ! time_usage = {time.time() - begin_time}")
    return dataloader


@exception_catcher
def model_selection(params: dict, args: Namespace):
    from src.logger import logger
    logger.info(f"begin run model_selection on UDF runtime with CPU only")

    begin = time.time()
    # logger.info(params["mini_batch"])

    mini_batch_data = json.loads(params["mini_batch"])
    budget = float(params["budget"])

    from src.eva_engine.run_ms import RunModelSelection

    dataloader = generate_dataloader(mini_batch_data=mini_batch_data, args=args)

    data_loader = [dataloader, dataloader, dataloader]

    logger.info(f"[end2end model_selection] Done with dataloader generation, time usage = " + str(time.time() - begin))

    begin = time.time()

    rms = RunModelSelection(args.search_space, args, is_simulate=args.is_simulate)
    best_arch, best_arch_performance, time_usage, _, p1_trace_highest_score, p1_trace_highest_scored_models_id = \
        rms.select_model_online_clean(
            budget=budget,
            data_loader=data_loader,
            only_phase1=False,
            run_workers=1)

    logger.info(f"[end2end model_selection] Done with model selection, time usage = " + str(time.time() - begin))

    # here is some response notation
    if best_arch_performance == 0:
        best_arch_performance_str = "Not Fully Train Yet"
    else:
        best_arch_performance_str = str(best_arch_performance)

    return orjson.dumps(
        {"best_arch": best_arch,
         "best_arch_performance": best_arch_performance_str,
         "time_usage": time_usage}).decode('utf-8')


@exception_catcher
def profiling_filtering_phase(params: dict, args: Namespace):
    from src.logger import logger
    logger.info(f"begin run profiling_filtering_phase CPU only")

    mini_batch_m = params["mini_batch"]

    from src.eva_engine.run_ms import RunModelSelection

    logger.info(f"begin run filtering phase at {os.getcwd()}, with {mini_batch_m}")

    mini_batch_data = json.loads(mini_batch_m)
    dataloader = generate_dataloader(mini_batch_data=mini_batch_data, args=args)
    data_loader = [dataloader, dataloader, dataloader]

    rms = RunModelSelection(args.search_space, args, is_simulate=args.is_simulate)
    score_time_per_model = rms.profile_filtering(data_loader=data_loader)

    return orjson.dumps({"time": score_time_per_model}).decode('utf-8')


@exception_catcher
def profiling_refinement_phase(params: dict, args: Namespace):
    from src.logger import logger
    logger.info(f"begin run profiling_refinement_phase CPU only")

    mini_batch_m = params["mini_batch"]

    from src.eva_engine.run_ms import RunModelSelection

    mini_batch_data = json.loads(mini_batch_m)

    dataloader = generate_dataloader(mini_batch_data=mini_batch_data, args=args)
    data_loader = [dataloader, dataloader, dataloader]

    rms = RunModelSelection(args.search_space, args, is_simulate=args.is_simulate)
    train_time_per_epoch = rms.profile_refinement(data_loader=data_loader)

    return orjson.dumps({"time": train_time_per_epoch}).decode('utf-8')


@exception_catcher
def coordinator(params: dict, args: Namespace):
    from src.logger import logger

    logger.info(f"begin run coordinator")

    budget = float(params["budget"])
    score_time_per_model = float(params["score_time_per_model"])
    train_time_per_epoch = float(params["train_time_per_epoch"])
    only_phase1 = True if params["only_phase1"].lower() == "true" else False

    from src.eva_engine.run_ms import RunModelSelection

    logger.info(f"coordinator params: budget={budget}, "
                f"score_time_per_model={score_time_per_model}, "
                f"train_time_per_epoch={train_time_per_epoch}, "
                f"only_phase1={only_phase1}")

    rms = RunModelSelection(args.search_space, args, is_simulate=args.is_simulate)
    K, U, N = rms.coordination(
        budget=budget,
        score_time_per_model=score_time_per_model,
        train_time_per_epoch=train_time_per_epoch,
        only_phase1=only_phase1)

    logger.info(f"coordinator done with K, U, N with {K, U, N}")

    return orjson.dumps(
        {"k": K, "u": U, "n": N}).decode('utf-8')


@exception_catcher
def filtering_phase(params: dict, args: Namespace):
    from src.logger import logger
    logger.info(f"begin run filtering_phase CPU only")

    # mini_batch_m = params["mini_batch"]
    n = int(params["n"])
    k = int(params["k"])

    from src.eva_engine.run_ms import RunModelSelection

    # mini_batch_data = json.loads(mini_batch_m)
    # dataloader = generate_dataloader(mini_batch_data=mini_batch_data, args=args)

    rms = RunModelSelection(args.search_space, args, is_simulate=args.is_simulate)
    k_models, _, _, _ = rms.filtering_phase(N=n, K=k)

    return orjson.dumps({"k_models": k_models}).decode('utf-8')


@exception_catcher
def filtering_phase_dataLoader(params: dict, args: Namespace):
    from src.logger import logger
    logger.info(f"begin run filtering_phase CPU only")

    mini_batch_m = params["mini_batch"]
    n = int(params["n"])
    k = int(params["k"])

    from src.eva_engine.run_ms import RunModelSelection

    mini_batch_data = json.loads(mini_batch_m)
    dataloader = generate_dataloader(mini_batch_data=mini_batch_data, args=args)

    rms = RunModelSelection(args.search_space, args, is_simulate=args.is_simulate)
    k_models, _, _, _ = rms.filtering_phase(N=n, K=k, train_loader=dataloader)

    return orjson.dumps({"k_models": k_models}).decode('utf-8')


@exception_catcher
def refinement_phase(params: dict, args: Namespace):
    mini_batch_m = params["mini_batch"]
    return orjson.dumps(
        {"k_models": "k_models"}).decode('utf-8')


@exception_catcher
def model_selection_workloads(params: dict, args: Namespace):
    """
    Run filtering (explore N models) and refinement phase (refine K models) for benchmarking latency.
    """

    mini_batch_m = params["mini_batch"]
    n = int(params["n"])
    k = int(params["k"])

    from src.logger import logger
    logger.info(f"begin run model_selection_workloads on CPU only, explore N={n} and K={k}")

    from src.eva_engine.run_ms import RunModelSelection

    mini_batch_data = json.loads(mini_batch_m)
    dataloader = generate_dataloader(mini_batch_data=mini_batch_data, args=args)
    rms = RunModelSelection(args.search_space, args, is_simulate=args.is_simulate)
    k_models, _, _, _ = rms.filtering_phase(N=n, K=k, train_loader=dataloader)
    best_arch, best_arch_performance, _, _ = rms.refinement_phase(
        U=1,
        k_models=k_models,
        train_loader=dataloader,
        valid_loader=dataloader)

    return orjson.dumps(
        {"best_arch": best_arch,
         "best_arch_performance": best_arch_performance,
         }).decode('utf-8')


@exception_catcher
def test_io(params: dict, args: Namespace):
    return orjson.dumps({"inputs are": json.dumps(params)}).decode('utf-8')


@exception_catcher
def model_selection_trails(params: dict, args: Namespace):
    from src.logger import logger
    logger.info(f"begin run model_selection_trails CPU  + GPU")

    mini_batch_data = json.loads(params["mini_batch"])
    budget = float(params["budget"])

    # 1. launch cache service
    columns = list(mini_batch_data[0].keys())
    requests.post(args.cache_svc_url,
                  json={'columns': columns, 'name_space': "train", 'table_name': "dummy",
                        "batch_size": len(mini_batch_data)})
    requests.post(args.cache_svc_url,
                  json={'columns': columns, 'name_space': "valid", 'table_name': "dummy",
                        "batch_size": len(mini_batch_data)})

    from src.eva_engine.run_ms import RunModelSelection

    # 2. profiling & coordination
    dataloader = generate_dataloader(mini_batch_data=mini_batch_data, args=args)
    data_loader = [dataloader, dataloader, dataloader]
    rms = RunModelSelection(args.search_space, args, is_simulate=args.is_simulate)

    begin_time = time.time()
    score_time_per_model = rms.profile_filtering(data_loader)
    train_time_per_epoch = rms.profile_refinement(data_loader)
    K, U, N = rms.coordination(budget, score_time_per_model, train_time_per_epoch, False)

    # 3. filtering
    k_models, all_models, p1_trace_highest_score, p1_trace_highest_scored_models_id = rms.filtering_phase(
        N, K, train_loader=data_loader[0])

    # 4. Run refinement pahse
    data = {'u': 1, 'k_models': k_models, "table_name": "dummy", "config_file": args.config_file}
    response = requests.post(args.refinement_url, json=data).json()

    best_arch, best_arch_performance = response["best_arch"], response["best_arch_performance"]

    end_time = time.time()
    real_time_usage = end_time - begin_time

    return orjson.dumps(
        {"best_arch": best_arch,
         "best_arch_performance": best_arch_performance,
         "time_usage": real_time_usage}).decode('utf-8')


@exception_catcher
def model_selection_trails_workloads(params: dict, args: Namespace):
    """
    Run filtering (explore N models) and refinement phase (refine K models) for benchmarking latency.
    """

    begin_time = time.time()
    mini_batch_data = json.loads(params["mini_batch"])
    n = int(params["n"])
    k = int(params["k"])

    # 1. launch cache service, for both train and valid.
    # todo: use real data table or others
    columns = list(mini_batch_data[0].keys())
    requests.post(args.cache_svc_url,
                  json={'columns': columns, 'name_space': "train", 'table_name': "dummy",
                        "batch_size": len(mini_batch_data)})
    requests.post(args.cache_svc_url,
                  json={'columns': columns, 'name_space': "valid", 'table_name': "dummy",
                        "batch_size": len(mini_batch_data)})

    from src.logger import logger
    logger.info(f"begin run model_selection_trails_workloads CPU + GPU, explore N={n} and K={k}")

    from src.eva_engine.run_ms import RunModelSelection

    # 2. filtering
    dataloader = generate_dataloader(mini_batch_data=mini_batch_data, args=args)
    rms = RunModelSelection(args.search_space, args, is_simulate=args.is_simulate)
    k_models, _, _, _ = rms.filtering_phase(N=n, K=k, train_loader=dataloader)

    # 3. Run refinement pahse
    data = {'u': 1, 'k_models': k_models, "table_name": "dummy", "config_file": args.config_file}
    response = requests.post(args.refinement_url, json=data).json()
    best_arch, best_arch_performance = response["best_arch"], response["best_arch_performance"]
    real_time_usage = time.time() - begin_time

    return orjson.dumps(
        {"best_arch": best_arch,
         "best_arch_performance": best_arch_performance,
         "time_usage": real_time_usage
         }).decode('utf-8')


# benchmarking code here
@exception_catcher
def benchmark_filtering_phase_latency(params: dict, args: Namespace):
    from src.logger import logger
    from src.common.structure import ModelAcquireData
    from src.controller.sampler_all.seq_sampler import SequenceSampler
    from src.eva_engine.phase1.evaluator import P1Evaluator
    from src.search_space.init_search_space import init_search_space
    from src.tools.io_tools import write_json, read_json
    from src.tools.res_measure import print_cpu_gpu_usage
    logger.info(f"begin run filtering_phase CPU only")

    args.models_explore = int(params["explore_models"])

    output_file = f"{args.result_dir}/score_{args.search_space}_{args.dataset}_batch_size_{args.batch_size}_{args.device}_{args.tfmem}.json"
    time_output_file = f"{args.result_dir}/time_score_{args.search_space}_{args.dataset}_batch_size_{args.batch_size}_{args.device}_{args.tfmem}.json"
    res_output_file = f"{args.result_dir}/resource_score_{args.search_space}_{args.dataset}_batch_size_{args.batch_size}_{args.device}_{args.tfmem}.json"

    # start the resource monitor
    stop_event, thread = print_cpu_gpu_usage(interval=0.5, output_file=res_output_file)

    db_config = {
        "db_name": args.db_name,
        "db_user": args.db_user,
        "db_host": args.db_host,
        "db_port": args.db_port,
    }

    search_space_ins = init_search_space(args)
    _evaluator = P1Evaluator(device=args.device,
                             num_label=args.num_labels,
                             dataset_name=args.dataset,
                             search_space_ins=search_space_ins,
                             train_loader=None,
                             is_simulate=False,
                             metrics=args.tfmem,
                             enable_cache=args.embedding_cache_filtering,
                             db_config=db_config)

    sampler = SequenceSampler(search_space_ins)
    explored_n = 0
    result = read_json(output_file)
    print(f"begin to score all, currently we already explored {len(result.keys())}")
    logger.info(f"begin to score all, currently we already explored {len(result.keys())}")

    while True:
        arch_id, arch_micro = sampler.sample_next_arch()
        if arch_id is None:
            break
        if arch_id in result:
            continue
        if explored_n > args.models_explore:
            break
        # run the model selection
        model_encoding = search_space_ins.serialize_model_encoding(arch_micro)
        model_acquire_data = ModelAcquireData(model_id=arch_id,
                                              model_encoding=model_encoding,
                                              is_last=False)
        data_str = model_acquire_data.serialize_model()
        model_score = _evaluator.p1_evaluate(data_str)
        explored_n += 1
        result[arch_id] = model_score
        if explored_n % 50 == 0:
            logger.info(f"Evaluate {explored_n} models")
            print(f"Evaluate {explored_n} models")

    if _evaluator.if_cuda_avaiable():
        torch.cuda.synchronize()

    # the first two are used for warming up
    _evaluator.time_usage["io_latency"] = \
        sum(_evaluator.time_usage["track_io_model_load"][2:]) + \
        sum(_evaluator.time_usage["track_io_model_init"][2:]) + \
        sum(_evaluator.time_usage["track_io_res_load"][2:]) + \
        sum(_evaluator.time_usage["track_io_data_retrievel"][2:]) + \
        sum(_evaluator.time_usage["track_io_data_preprocess"][2:])

    _evaluator.time_usage["compute_latency"] = sum(_evaluator.time_usage["track_compute"][2:])
    _evaluator.time_usage["latency"] = _evaluator.time_usage["io_latency"] + _evaluator.time_usage["compute_latency"]

    _evaluator.time_usage["avg_compute_latency"] = \
        _evaluator.time_usage["compute_latency"] \
        / len(_evaluator.time_usage["track_compute"][2:])

    write_json(output_file, result)
    # compute time
    write_json(time_output_file, _evaluator.time_usage)

    # Then, at the end of your program, you can stop the thread:
    print("Done, time sleep for 10 seconds")
    # wait the resource montor flush
    time.sleep(10)
    stop_event.set()
    thread.join()

    return orjson.dumps({"Write to": time_output_file}).decode('utf-8')


# Micro benchmarking filterting phaes
search_space_ins = None
_evaluator = None
sampler = None


@exception_catcher
def in_db_filtering_state_init(params: dict, args: Namespace):
    global search_space_ins, _evaluator, sampler
    from src.logger import logger
    from src.controller.sampler_all.seq_sampler import SequenceSampler
    from src.eva_engine.phase1.evaluator import P1Evaluator
    from src.search_space.init_search_space import init_search_space

    db_config = {
        "db_name": args.db_name,
        "db_user": args.db_user,
        "db_host": args.db_host,
        "db_port": args.db_port,
    }

    # init once
    # params["eva_results"] == "null" means it a new job
    if params["eva_results"] == "null" or (search_space_ins is None and _evaluator is None and sampler is None):
        logger.info(f'New job = {params["eva_results"]}, search_space_ins = {search_space_ins}')
        search_space_ins = init_search_space(args)
        _evaluator = P1Evaluator(device=args.device,
                                 num_label=args.num_labels,
                                 dataset_name=params["dataset"],
                                 search_space_ins=search_space_ins,
                                 train_loader=None,
                                 is_simulate=False,
                                 metrics=args.tfmem,
                                 enable_cache=args.embedding_cache_filtering,
                                 db_config=db_config,
                                 data_retrievel="spi")
        sampler = SequenceSampler(search_space_ins)

    arch_id, arch_micro = sampler.sample_next_arch()
    model_encoding = search_space_ins.serialize_model_encoding(arch_micro)

    return orjson.dumps({"model_encoding": model_encoding, "arch_id": arch_id}).decode('utf-8')


@exception_catcher
def in_db_filtering_evaluate(params: dict, args: Namespace):
    global search_space_ins, _evaluator, sampler
    from src.common.structure import ModelAcquireData
    from src.logger import logger
    try:
        if search_space_ins is None and _evaluator is None and sampler is None:
            logger.info("search_space_ins, _evaluator, sampler is None")
            return orjson.dumps({"error": "erroed, plz call init first"}).decode('utf-8')

        begin_read = time.time()
        mini_batch = get_data_from_shared_memory_int(int(params["rows"]))
        read_done = time.time()
        # logger.info(mini_batch)
        # logger.info(mini_batch.size())
        # logger.info(list(mini_batch[0]))

        logger.info(f"Data Retrievel time {params['spi_seconds']}, "
                    f"read shared memory time = {read_done - begin_read}")

        sampled_result = json.loads(params["sample_result"])
        arch_id, model_encoding = str(sampled_result["arch_id"]), str(sampled_result["model_encoding"])

        logger.info(f"Begin evaluate {params['model_index']}, "
                    f"with size of batch = {len(mini_batch)}, "
                    f"size of columns = {len(mini_batch[0])}")
        model_acquire_data = ModelAcquireData(model_id=arch_id,
                                              model_encoding=model_encoding,
                                              is_last=False,
                                              spi_seconds=float(params["spi_seconds"]) + read_done - begin_read,
                                              spi_mini_batch=mini_batch,
                                              batch_size=int(params["rows"])
                                              )

        model_score = _evaluator._p1_evaluate_online(model_acquire_data)
        logger.info(f'Done evaluate {params["model_index"]}, '
                    f'with {orjson.dumps({"index": params["model_index"], "score": model_score}).decode("utf-8")}')
    except:
        logger.info(orjson.dumps(
            {"Errored": traceback.format_exc()}).decode('utf-8'))

        return orjson.dumps(
            {"Errored": traceback.format_exc()}).decode('utf-8')

    return orjson.dumps({"index": params["model_index"], "score": model_score}).decode('utf-8')


@exception_catcher
def records_results(params: dict, args: Namespace):
    global search_space_ins, _evaluator, sampler
    from src.tools.io_tools import write_json
    from src.logger import logger

    try:
        time_output_file = f"{args.result_dir}/time_score_{args.search_space}_{params['dataset']}_batch_size_{args.batch_size}_{args.device}_{args.tfmem}.json"
        _evaluator.time_usage["io_latency"] = \
            sum(_evaluator.time_usage["track_io_model_load"][2:]) + \
            sum(_evaluator.time_usage["track_io_model_init"][2:]) + \
            sum(_evaluator.time_usage["track_io_res_load"][2:]) + \
            sum(_evaluator.time_usage["track_io_data_retrievel"][2:]) + \
            sum(_evaluator.time_usage["track_io_data_preprocess"][2:])

        _evaluator.time_usage["compute_latency"] = sum(_evaluator.time_usage["track_compute"][2:])
        _evaluator.time_usage["latency"] = _evaluator.time_usage["io_latency"] + _evaluator.time_usage[
            "compute_latency"]

        _evaluator.time_usage["avg_compute_latency"] = \
            _evaluator.time_usage["compute_latency"] \
            / len(_evaluator.time_usage["track_compute"][2:])

        logger.info(f"Saving time usag to {time_output_file}")
        # compute time
        write_json(time_output_file, _evaluator.time_usage)
    except:
        logger.info(orjson.dumps(
            {"Errored": traceback.format_exc()}).decode('utf-8'))

        return orjson.dumps(
            {"Errored": traceback.format_exc()}).decode('utf-8')

    return orjson.dumps({"Done": 1}).decode('utf-8')


@exception_catcher
def measure_call_overheads(params: dict, args: Namespace):
    return orjson.dumps({"Done": 1}).decode('utf-8')


import numpy as np
from multiprocessing import shared_memory


def get_data_from_shared_memory_int(n_rows):
    shm = shared_memory.SharedMemory(name="my_shared_memory")
    data = np.frombuffer(shm.buf, dtype=np.float32)
    data = data.reshape(n_rows, -1)
    return data


if __name__ == "__main__":
    params = {}
    params["budget"] = 10
    params["score_time_per_model"] = 0.0211558125
    params["train_time_per_epoch"] = 5.122203075885773
    params["only_phase1"] = 'true'
    params["config_file"] = './internal/ml/model_selection/config.ini'
    print(coordinator(json.dumps(params)))

    params = {}
    params[
        "mini_batch"] = '[{"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"1"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}, {"col1":"123:123","col2":"123:123","col3":"123:123","label":"0"}]'
    params["n"] = 10
    params["k"] = 1
    params["config_file"] = './internal/ml/model_selection/config.ini'
    print(filtering_phase_dataLoader(json.dumps(params)))
