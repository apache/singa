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


# this is for checking the flops and params
try:
    from thop import profile
except:
    pass
from src.common.constant import Config, CommonVars
from src.common.structure import ModelAcquireData
from src.eva_engine import evaluator_register
from src.query_api.interface import SimulateScore
from src.dataset_utils import dataset
from torch.utils.data import DataLoader
import torch
import time
from torch import nn
from src.search_space.core.space import SpaceWrapper
import psycopg2
from typing import Any, List, Dict, Tuple
from src.logger import logger


class P1Evaluator:

    def __init__(self, device: str, num_label: int, dataset_name: str,
                 search_space_ins: SpaceWrapper,
                 train_loader: DataLoader, is_simulate: bool, metrics: str = CommonVars.ExpressFlow,
                 enable_cache: bool = False, db_config: Dict = None,
                 data_retrievel: str = "sql"):
        """
        :param device:
        :param num_label:
        :param dataset_name:
        :param search_space_ins:
        :param search_space_ins:
        :param train_loader:
        :param is_simulate:
        :param metrics: which TFMEM to use?
        :param enable_cache: if cache embedding for scoring? only used on structued data
        :param db_config: how to connect to databaes
        :param data_retrievel: sql or spi
        """
        self.metrics = metrics
        self.is_simulate = is_simulate
        # used only is_simulate = True
        self.score_getter = None

        # dataset settings
        self.dataset_name = dataset_name
        self.train_loader = train_loader
        self.num_labels = num_label

        self.search_space_ins = search_space_ins

        self.device = device

        # this is to do the expeirment
        self.enable_cache = enable_cache
        self.model_cache = None

        # performance records
        self.time_usage = {
            "model_id": [],

            "latency": 0.0,
            "io_latency": 0.0,
            "compute_latency": 0.0,

            "track_compute": [],  # compute time
            "track_io_model_init": [],  # init model weight
            "track_io_model_load": [],  # load model into GPU/CPU
            "track_io_res_load": [],  # load result into GPU/CPU
            "track_io_data_retrievel": [],  # release data
            "track_io_data_preprocess": [],  # pre-processing
        }

        self.db_config = db_config
        self.last_id = -1
        self.data_retrievel = data_retrievel

        # at the benchmarking, we only use one batch for fast evaluate
        self.cached_mini_batch = None
        self.cached_mini_batch_target = None

        self.conn = None

    def if_cuda_avaiable(self):
        if "cuda" in self.device:
            return True
        else:
            return False

    def p1_evaluate(self, data_str: dict) -> dict:
        """
        :param data_str: encoded ModelAcquireData
        :return:
        """

        model_acquire = ModelAcquireData.deserialize(data_str)

        if self.is_simulate:
            if self.metrics == "jacflow":
                return self._p1_evaluate_simu_jacflow(model_acquire)
            else:
                return self._p1_evaluate_simu(model_acquire)
        else:
            return self._p1_evaluate_online(model_acquire)

    def measure_model_flops(self, data_str: dict, batch_size: int, channel_size: int):
        # todo: check the package
        mini_batch, mini_batch_targets, _ = self.retrievel_data(None)
        model_acquire = ModelAcquireData.deserialize(data_str)
        model_encoding = model_acquire.model_encoding
        new_model = self.search_space_ins.new_arch_scratch_with_default_setting(model_encoding, bn=True)
        if self.search_space_ins.name == Config.MLPSP:
            new_model.init_embedding(requires_grad=True)
        new_model = new_model.to(self.device)
        flops, params = profile(new_model, inputs=(mini_batch,))
        print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
        print('Params = ' + str(params / 1000 ** 2) + 'M')

    # # 1. Score NasWot
    # new_model = self.search_space_ins.new_arch_scratch_with_default_setting(model_encoding, bn=True)
    # new_model = new_model.to(self.device)
    # naswot_score, _ = evaluator_register[CommonVars.NAS_WOT].evaluate_wrapper(
    #     arch=new_model,
    #     device=self.device,
    #     space_name = self.search_space_ins.name,
    #     batch_data=self.mini_batch,
    #     batch_labels=self.mini_batch_targets)
    #
    # # 2. Score SynFlow
    # new_model = self.search_space_ins.new_arch_scratch_with_default_setting(model_encoding, bn=False)
    # new_model = new_model.to(self.device)
    # synflow_score, _ = evaluator_register[CommonVars.PRUNE_SYNFLOW].evaluate_wrapper(
    #     arch=new_model,
    #     device=self.device,
    #     space_name = self.search_space_ins.name,
    #     batch_data=self.mini_batch,
    #     batch_labels=self.mini_batch_targets)
    #
    # # 3. combine the result and return
    # model_score = {CommonVars.NAS_WOT: naswot_score,
    #                CommonVars.PRUNE_SYNFLOW: synflow_score}

    def _p1_evaluate_online(self, model_acquire: ModelAcquireData) -> dict:

        model_encoding = model_acquire.model_encoding

        # 1. Get a batch of data
        mini_batch, mini_batch_targets, data_load_time_usage, data_pre_process_time = self.retrievel_data(model_acquire)
        # logger.info(
        #     f"mini_batch sizes - id: {mini_batch['id'].size()}, value: {mini_batch['value'].size()},
        #     targets: {mini_batch_targets.size()}")
        # print(
        #     f"mini_batch sizes - id: {mini_batch['id'].size()}, value: {mini_batch['value'].size()},
        #     targets: {mini_batch_targets.size()}")
        self.time_usage["track_io_data_retrievel"].append(data_load_time_usage)

        # 2. Score all tfmem
        if self.metrics == CommonVars.ALL_EVALUATOR:
            model_score = {}
            for alg, score_evaluator in evaluator_register.items():
                if alg == CommonVars.PRUNE_SYNFLOW or alg == CommonVars.ExpressFlow:
                    bn = False
                else:
                    bn = True
                new_model = self.search_space_ins.new_arch_scratch_with_default_setting(model_encoding, bn=bn)
                if self.search_space_ins.name == Config.MLPSP:
                    new_model.init_embedding()
                new_model = new_model.to(self.device)

                mini_batch = self.data_pre_processing(mini_batch, self.metrics, new_model)

                _score, _ = score_evaluator.evaluate_wrapper(
                    arch=new_model,
                    device=self.device,
                    space_name=self.search_space_ins.name,
                    batch_data=mini_batch,
                    batch_labels=mini_batch_targets)

                _score = _score.item()
                model_score[alg] = abs(_score)

                # clear the cache
                if "cuda" in self.device:
                    torch.cuda.empty_cache()

        elif self.metrics == CommonVars.JACFLOW:
            begin = time.time()
            new_model = self.search_space_ins.new_arch_scratch_with_default_setting(model_encoding, bn=False)
            if self.search_space_ins.name == Config.MLPSP:
                if self.enable_cache:
                    new_model.init_embedding(self.model_cache)
                    if self.model_cache is None:
                        self.model_cache = new_model.embedding.to(self.device)
                else:
                    # init embedding every time created a new model
                    new_model.init_embedding()
            time_usage = time.time() - begin
            self.time_usage["track_io_model_init"].append(time_usage)
            print("Model Init",  self.enable_cache, time_usage)

            if self.if_cuda_avaiable():
                begin = time.time()
                new_model = new_model.to(self.device)
                torch.cuda.synchronize()
                self.time_usage["track_io_model_load"].append(time.time() - begin)
            else:
                self.time_usage["track_io_model_load"].append(0)

            # measure data load time
            begin = time.time()
            all_one_mini_batch = self.data_pre_processing(mini_batch, CommonVars.PRUNE_SYNFLOW, new_model)
            self.time_usage["track_io_data_preprocess"].append(data_pre_process_time + time.time() - begin)
            if self.search_space_ins.name == Config.MLPSP:
                print("compute with done", all_one_mini_batch.size(), mini_batch["id"].size(), mini_batch["value"].size())
                logger.info(
                    f"mini_batch sizes - {all_one_mini_batch.size()} "
                    f"id: {mini_batch['id'].size()}, value: {mini_batch['value'].size()},"
                    f"targets: {mini_batch_targets.size()}")

            _score_1, compute_time1 = evaluator_register[CommonVars.PRUNE_SYNFLOW].evaluate_wrapper(
                arch=new_model,
                device=self.device,
                space_name=self.search_space_ins.name,
                batch_data=all_one_mini_batch,
                batch_labels=mini_batch_targets)

            _score_2, compute_time2 = evaluator_register[CommonVars.NAS_WOT].evaluate_wrapper(
                arch=new_model,
                device=self.device,
                space_name=self.search_space_ins.name,
                batch_data=mini_batch,
                batch_labels=mini_batch_targets)
            print(compute_time1, compute_time2)
            logger.info(f"{compute_time1}, {compute_time2}")

            self.time_usage["track_compute"].append(compute_time1 + compute_time2)
            self.time_usage["model_id"].append(model_encoding)

            if self.if_cuda_avaiable():
                begin = time.time()
                _score = _score_1.item() + _score_2
                torch.cuda.synchronize()
                self.time_usage["track_io_res_load"].append(time.time() - begin)
            else:
                _score = _score_1.item() + _score_2
                self.time_usage["track_io_res_load"].append(0)

            model_score = {self.metrics: float(abs(_score))}
            del new_model
        # 2. score using only one metrics
        else:
            if self.metrics == CommonVars.PRUNE_SYNFLOW or self.metrics == CommonVars.ExpressFlow:
                bn = False
            else:
                bn = True
            # measure model load time
            begin = time.time()
            new_model = self.search_space_ins.new_arch_scratch_with_default_setting(model_encoding, bn=bn)

            # # mlp have embedding layer, which can be cached, optimization!
            # if self.search_space_ins.name == Config.MLPSP:
            #     if self.enable_cache:
            #         new_model.init_embedding(self.model_cache)
            #         if self.model_cache is None:
            #             self.model_cache = new_model.embedding.to(self.device)
            #     else:
            #         # init embedding every time created a new model
            #         new_model.init_embedding()

            self.time_usage["track_io_model_init"].append(time.time() - begin)

            if self.if_cuda_avaiable():
                begin = time.time()
                new_model = new_model.to(self.device)
                torch.cuda.synchronize()
                self.time_usage["track_io_model_load"].append(time.time() - begin)
            else:
                self.time_usage["track_io_model_load"].append(0)

            # measure data load time
            begin = time.time()
            mini_batch = self.data_pre_processing(mini_batch, self.metrics, new_model)
            self.time_usage["track_io_data_preprocess"].append(data_pre_process_time + time.time() - begin)

            _score, compute_time = evaluator_register[self.metrics].evaluate_wrapper(
                arch=new_model,
                device=self.device,
                space_name=self.search_space_ins.name,
                batch_data=mini_batch,
                batch_labels=mini_batch_targets)

            self.time_usage["track_compute"].append(compute_time)

            if self.if_cuda_avaiable():
                begin = time.time()
                _score = _score.item()
                torch.cuda.synchronize()
                self.time_usage["track_io_res_load"].append(time.time() - begin)

            else:
                _score = _score.item()
                self.time_usage["track_io_res_load"].append(0)

            model_score = {self.metrics: abs(_score)}
            del new_model
        return model_score

    def _p1_evaluate_simu_jacflow(self, model_acquire: ModelAcquireData) -> dict:
        """
        This involves get rank, and get jacflow
        """
        if self.score_getter is None:
            self.score_getter = SimulateScore(space_name=self.search_space_ins.name,
                                              dataset_name=self.dataset_name)

        model_score = self.score_getter.query_tfmem_rank_score(arch_id=model_acquire.model_id)

        return model_score

    def _p1_evaluate_simu(self, model_acquire: ModelAcquireData) -> dict:
        """
        This involves simulate get alls core,
        """
        if self.score_getter is None:
            self.score_getter = SimulateScore(space_name=self.search_space_ins.name,
                                              dataset_name=self.dataset_name)

        score = self.score_getter.query_all_tfmem_score(arch_id=model_acquire.model_id)
        model_score = {self.metrics: abs(float(score[self.metrics]))}
        return model_score

    def retrievel_data(self, model_acquire):
        if not self.is_simulate:
            if self.dataset_name in [Config.c10, Config.c100, Config.imgNet, Config.imgNetFull]:
                if self.train_loader is None:
                    raise f"self.train_loader is None for {self.dataset_name}"
                # for img data
                begin = time.time()
                mini_batch, mini_batch_targets = dataset.get_mini_batch(
                    dataloader=self.train_loader,
                    sample_alg="random",
                    batch_size=model_acquire.batch_size,
                    num_classes=self.num_labels)
                mini_batch.to(self.device)
                mini_batch_targets.to(self.device)
                # wait for moving data to GPU
                if self.if_cuda_avaiable():
                    torch.cuda.synchronize()
                time_usage = time.time() - begin
                # todo: here is inaccurate
                return mini_batch, mini_batch_targets, time_usage, 0
            elif self.dataset_name in [Config.Criteo, Config.Frappe, Config.UCIDataset]:
                if self.train_loader is None:
                    if self.data_retrievel == "sql":
                        batch, time_usage = self._retrievel_from_db_sql(model_acquire.batch_size)
                        data_tensor, y_tensor, process_time = self.sql_batch_data_pre_processing(batch)
                        return data_tensor, y_tensor, time_usage, process_time
                    elif self.data_retrievel == "spi":
                        batch, time_usage = self._retrievel_from_db_spi(model_acquire)
                        # pre-processing
                        begin = time.time()
                        id_tensor = torch.LongTensor(batch[:, 1::2]).to(self.device)
                        value_tensor = torch.FloatTensor(batch[:, 2::2]).to(self.device)
                        y_tensor = torch.FloatTensor(batch[:, 0:1]).to(self.device)
                        data_tensor = {'id': id_tensor, 'value': value_tensor, 'y': y_tensor}
                        logger.info(id_tensor.size())
                        return data_tensor, y_tensor, time_usage + time.time() - begin, 0
                else:
                    if self.cached_mini_batch is None and self.cached_mini_batch_target is None:
                        # this is structure data
                        begin = time.time()
                        batch = iter(self.train_loader).__next__()
                        target = batch['y'].type(torch.LongTensor).to(self.device)
                        batch['id'] = batch['id'].to(self.device)
                        batch['value'] = batch['value'].to(self.device)

                        # wait for moving data to GPU
                        if self.if_cuda_avaiable():
                            torch.cuda.synchronize()
                        time_usage = time.time() - begin
                        self.cached_mini_batch = batch
                        self.cached_mini_batch_target = target
                        return batch, target, time_usage, 0
                    else:
                        return self.cached_mini_batch, self.cached_mini_batch_target, 0, 0
            else:
                # here is to test the expressflow
                # todo: debut, this is to debut, mannuly tune it
                y_tensor = torch.rand(1)
                dimensions = 2000
                data_tensor = {'id': torch.rand([1, dimensions]), 'value': torch.rand([1, dimensions]), 'y': y_tensor}
                return data_tensor, y_tensor, 0, 0

    def connect_to_db(self):
        try:
            self.conn = psycopg2.connect(
                dbname=self.db_config["db_name"],
                user=self.db_config["db_user"],
                host=self.db_config["db_host"],
                port=self.db_config["db_port"]
            )
        except Exception as e:
            print(f"Error connecting to the database: {e}")

    def _retrievel_from_db_sql(self, batch_size):

        begin_time = time.time()
        if self.conn is None or self.conn.closed:
            # If the connection is not established or was closed, reconnect.
            self.connect_to_db()

        # fetch and preprocess data from PostgreSQL
        cur = self.conn.cursor()

        cur.execute(f"SELECT * FROM {self.dataset_name}_train WHERE id > {self.last_id} LIMIT {batch_size};")
        rows = cur.fetchall()

        if self.last_id <= 80000:
            # Update last_id with max id of fetched rows
            self.last_id = max(row[0] for row in rows)  # assuming 'id' is at index 0
        else:
            # If no more new rows, reset last_id to start over scan and return 'end_position'
            self.last_id = 0

        # block until a free slot is available
        time_usage = time.time() - begin_time
        return rows, time_usage

    def _retrievel_from_db_spi(self, model_acquire):
        batch = model_acquire.spi_mini_batch
        data_retrieval_time_usage = model_acquire.spi_seconds
        return batch, data_retrieval_time_usage

    def data_pre_processing(self, mini_batch, metrics: str, new_model: nn.Module):

        # for those two metrics, we use all one embedding for efficiency (as in their paper)
        if metrics in [CommonVars.ExpressFlow, CommonVars.PRUNE_SYNFLOW]:
            if isinstance(mini_batch, torch.Tensor):
                feature_dim = list(mini_batch[0, :].shape)
                # add one dimension to feature dim, [1] + [3, 32, 32] = [1, 3, 32, 32]
                mini_batch = torch.ones([1] + feature_dim).float().to(self.device)
            else:
                # this is for the tabular data,
                mini_batch = new_model.generate_all_ones_embedding().float().to(self.device)
                # print(mini_batch.size())
        else:
            # for others, skip preprocessing
            pass

        # wait for moving data to GPU
        if self.if_cuda_avaiable():
            torch.cuda.synchronize()
        return mini_batch

    def sql_batch_data_pre_processing(self, queryed_rows: List[Tuple]):
        """
        mini_batch_data: [('0', '0', '123:123', '123:123', '123:123',)
        """

        # def decode_libsvm(columns):
        #     # Decode without additional mapping or zipping, directly processing the splits.
        #     ids = []
        #     values = []
        #     for col in columns[2:]:
        #         id, value = col.split(':')
        #         ids.append(int(id))
        #         values.append(float(value))
        #     return {'id': ids, 'value': values, 'y': int(columns[1])}

        def decode_libsvm(columns):
            map_func = lambda pair: (int(pair[0]), float(pair[1]))
            # 0 is id, 1 is label
            id, value = zip(*map(lambda col: map_func(col.split(':')), columns[2:]))
            sample = {'id': list(id),
                      'value': list(value),
                      'y': int(columns[1])}
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
                sample = decode_libsvm(list(row_value))
                feat_id.append(sample['id'])
                feat_value.append(sample['value'])
                y.append(sample['y'])
            return {'id': feat_id, 'value': feat_value, 'y': y}

        begin = time.time()
        batch = pre_processing(queryed_rows)
        id_tensor = torch.LongTensor(batch['id']).to(self.device)
        value_tensor = torch.FloatTensor(batch['value']).to(self.device)
        y_tensor = torch.FloatTensor(batch['y']).to(self.device)
        data_tensor = {'id': id_tensor, 'value': value_tensor, 'y': y_tensor}
        # wait for moving data to GPU
        if self.if_cuda_avaiable():
            torch.cuda.synchronize()
        duration = time.time() - begin
        return data_tensor, y_tensor, duration
