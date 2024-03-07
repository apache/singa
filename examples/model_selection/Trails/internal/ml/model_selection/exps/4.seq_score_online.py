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
from exps.shared_args import parse_arguments


def generate_data_loader():
    if args.dataset in [Config.c10, Config.c100, Config.imgNet, Config.imgNetFull]:
        train_loader, val_loader, class_num = dataset.get_dataloader(
            train_batch_size=args.batch_size,
            test_batch_size=args.batch_size,
            dataset=args.dataset,
            num_workers=1,
            datadir=os.path.join(args.base_dir))
        test_loader = val_loader
    else:
        train_loader, val_loader, test_loader = libsvm_dataloader(
            args=args,
            data_dir=os.path.join(args.base_dir, args.dataset),
            nfield=args.nfield,
            batch_size=args.batch_size)
        class_num = args.num_labels

    return train_loader, val_loader, test_loader, class_num


if __name__ == "__main__":
    args = parse_arguments()
    from src.common.constant import Config

    # set the log name
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)
    os.environ.setdefault("log_logger_folder_name", f"{args.log_folder}")
    os.environ.setdefault("log_file_name", args.log_name + "_" + str(ts) + ".log")
    os.environ.setdefault("base_dir", args.base_dir)

    from src.common.structure import ModelAcquireData
    from src.controller.sampler_all.seq_sampler import SequenceSampler
    from src.eva_engine.phase1.evaluator import P1Evaluator
    from src.logger import logger
    from src.search_space.init_search_space import init_search_space
    from src.dataset_utils.structure_data_loader import libsvm_dataloader
    from src.tools.io_tools import write_json, read_json
    from src.dataset_utils import dataset
    from src.common.constant import Config, CommonVars

    search_space_ins = init_search_space(args)

    train_loader, val_loader, test_loader, class_num = generate_data_loader()

    _evaluator = P1Evaluator(device=args.device,
                             num_label=args.num_labels,
                             dataset_name=args.dataset,
                             search_space_ins=search_space_ins,
                             train_loader=train_loader,
                             is_simulate=False,
                             metrics=args.tfmem,
                             enable_cache=args.embedding_cache_filtering)

    sampler = SequenceSampler(search_space_ins)

    explored_n = 0
    output_file = f"{args.result_dir}/score_{args.search_space}_{args.dataset}_batch_size_{args.batch_size}_{args.device}.json"
    result = read_json(output_file)
    print(f"begin to score all, currently we already explored {len(result.keys())}")
    logger.info(f"begin to score all, currently we already explored {len(result.keys())}")
    while True:
        arch_id, arch_micro = sampler.sample_next_arch()
        if arch_id is None:
            logger.info("Stop exploring, meet None arch id")
            break
        if arch_id in result:
            continue
        if args.models_explore != -1 and explored_n > args.models_explore:
            logger.info(f"Stop exploring, {explored_n} > {args.models_explore}")
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
        # print(f" {datetime.now()} finish arch = {arch_id}, model_score = {model_score}")

        if explored_n < 10:
            print("3. [trails] Phase 1: filter phase explored " + str(explored_n)
                  + "Total explored " + str(len(result)) +
                  " model, model_id = " + str(arch_id) +
                  " model_scores = " + json.dumps(model_score))
            logger.info("3. [trails] Phase 1: filter phase explored " + str(explored_n)
                        + "Total explored " + str(len(result)) +
                        " model, model_id = " + str(arch_id) +
                        " model_scores = " + json.dumps(model_score))
        if explored_n % 1000 == 0:
            # print_memory_usg()
            # _evaluator.force_gc()
            print("3. [trails] Phase 1: filter phase explored " + str(explored_n)
                  + "Total explored " + str(len(result)) +
                  " model, model_id = " + str(arch_id) +
                  " model_scores = " + json.dumps(model_score))
            logger.info("3. [trails] Phase 1: filter phase explored " + str(explored_n)
                        + "Total explored " + str(len(result)) +
                        " model, model_id = " + str(arch_id) +
                        " model_scores = " + json.dumps(model_score))
        if explored_n % 1000 == 0:
            # print_memory_usg()
            # _evaluator.force_gc()
            logger.info("3. [trails] Phase 1: filter phase explored " + str(explored_n) +
                        " model, model_id = " + str(arch_id) +
                        " model_scores = " + json.dumps(model_score))
            write_json(output_file, result)
    write_json(output_file, result)
