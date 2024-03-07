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
import json
import os
import time
import traceback
from singa import device as singa_device
import numpy as np

from exps.shared_args import parse_arguments

if __name__ == "__main__":

    args = parse_arguments()

    # set the log name
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)

    os.environ.setdefault("log_logger_folder_name", f"{args.log_folder}")
    os.environ.setdefault("log_file_name", f"{args.log_name}_{args.dataset}_ep{args.epoch}_{ts}.log")
    os.environ.setdefault("base_dir", args.base_dir)

    from src.logger import logger
    from src.eva_engine.phase2.algo.trainer import ModelTrainer
    from src.search_space.init_search_space import init_search_space
    from src.dataset_utils.structure_data_loader import libsvm_dataloader

    search_space_ins = init_search_space(args)
    search_space_ins.load()

    try:
        # read the checkpoint
        checkpoint_file_name = f"{args.result_dir}/train_config_tune_{args.dataset}_epo_{args.epoch}.json"

        # 1. data loader
        train_loader, val_loader, test_loader = libsvm_dataloader(
            args=args,
            data_dir=os.path.join(args.base_dir, args.dataset),
            nfield=args.nfield,
            batch_size=args.batch_size)

        arch_id = "128-128-128-128"
        print(f"begin to train the {arch_id}")

        model = search_space_ins.new_architecture(arch_id)
        if args.device == 'cpu':
            dev = singa_device.get_default_device()
        else:  # GPU
            dev = singa_device.create_cuda_gpu_on(args.local_rank)  # need to change to CPU device for CPU-only machines
        dev.SetRandSeed(0)
        np.random.seed(0)

        valid_auc, total_run_time, train_log = ModelTrainer.fully_train_arch(
            model=model,
            use_test_acc=False,
            epoch_num=args.epoch,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            args=args)

        logger.info(f' ----- model id: {arch_id}, Val_AUC : {valid_auc} Total running time: '
                    f'{total_run_time}-----')
        print(f' ----- model id: {arch_id}, Val_AUC : {valid_auc} Total running time: '
              f'{total_run_time}-----')

        # update the shared model eval res
        logger.info(f" ---- info: {json.dumps({arch_id: train_log})}")

        print(f" ---- info: {json.dumps({arch_id: train_log})}")

        logger.info(f" Saving result to: {checkpoint_file_name}")
    except:
        print(traceback.format_exc())
        logger.info(traceback.format_exc())
