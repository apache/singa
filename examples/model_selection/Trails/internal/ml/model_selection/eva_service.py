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
import argparse
import configparser
from sanic import Sanic
from sanic.exceptions import InvalidUsage
from sanic.response import json

ts = calendar.timegm(time.gmtime())
os.environ.setdefault("log_logger_folder_name", "log_eval_service")
os.environ.setdefault("log_file_name", "eval_service_" + str(ts) + ".log")
from src.logger import logger
from src.eva_engine.run_ms import RunModelSelection
from src.dataset_utils.stream_dataloader import StreamingDataLoader
from shared_config import parse_config_arguments
from typing import Any, List, Dict, Tuple


def refinement_phase(u: int, k_models: List, dataset_name: str, config_file: str):
    """
    U: training-epoches
    K-Models: k models to train
    config_file: config file path
    """
    args = parse_config_arguments(config_file)
    args.device = "cuda:7"
    train_dataloader = StreamingDataLoader(
        cache_svc_url=args.cache_svc_url, table_name=f"{dataset_name}_train", name_space="train")
    eval_dataloader = StreamingDataLoader(
        cache_svc_url=args.cache_svc_url, table_name=f"{dataset_name}_valid", name_space="valid")

    try:
        rms = RunModelSelection(args.search_space, args, is_simulate=args.is_simulate)
        best_arch, best_arch_performance, _, _ = rms.refinement_phase(
            U=u,
            k_models=k_models,
            train_loader=train_dataloader,
            valid_loader=eval_dataloader)
    finally:
        train_dataloader.stop()
        eval_dataloader.stop()
    return {"best_arch": best_arch, "best_arch_performance": best_arch_performance}


app = Sanic("evaApp")


@app.route("/", methods=["POST"])
async def start_refinement_phase(request):
    # Check if request is JSON
    if not request.json:
        logger.info("Expecting JSON payload")
        raise InvalidUsage("Expecting JSON payload")

    u = request.json.get('u')
    k_models = request.json.get('k_models')
    dataset_name = request.json.get('dataset_name')
    config_file = request.json.get('config_file')

    if u is None or k_models is None or config_file is None:
        logger.info(f"Missing 'u' or 'k_models' in JSON payload, {request.json}")
        raise InvalidUsage("Missing 'u' or 'k_models' in JSON payload")

    result = refinement_phase(u, k_models, dataset_name, config_file)

    return json(result)


if __name__ == "__main__":
    result = refinement_phase(
        u=1,
        k_models=["8-8-8-8", "16-16-16-16"],
        dataset_name="frappe",
        config_file="/home/xingnaili/firmest_docker/TRAILS/internal/ml/model_selection/config.ini")

    # app.run(host="0.0.0.0", port=8095)
