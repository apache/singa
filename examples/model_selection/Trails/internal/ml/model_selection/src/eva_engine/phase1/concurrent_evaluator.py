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
from multiprocessing import Manager
import gc


class ConcurrentP1Evaluator:

    def __init__(self, device: str, num_label: int, dataset_name: str,
                 search_space_ins: SpaceWrapper,
                 train_loader: DataLoader, is_simulate: bool, metrics: str = CommonVars.ExpressFlow,
                 enable_cache: bool = False):
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
        """
        self.metrics = metrics
        self.is_simulate = is_simulate

        self.dataset_name = dataset_name

        self.search_space_ins = search_space_ins

        self.device = device
        self.num_labels = num_label

        self.score_getter = None

        # get one mini batch
        if not self.is_simulate:
            if self.dataset_name in [Config.c10, Config.c100, Config.imgNet]:
                # for img data
                self.mini_batch, self.mini_batch_targets = dataset.get_mini_batch(
                    dataloader=train_loader,
                    sample_alg="random",
                    batch_size=32,
                    num_classes=self.num_labels)
                self.mini_batch.to(self.device)
                self.mini_batch_targets.to(self.device)
            elif self.dataset_name in [Config.Criteo, Config.Frappe, Config.UCIDataset]:
                # this is structure data
                batch = iter(train_loader).__next__()
                target = batch['y'].type(torch.LongTensor).to(self.device)
                batch['id'] = batch['id'].to(self.device)
                batch['value'] = batch['value'].to(self.device)
                self.mini_batch = batch
                self.mini_batch_targets = target.to(self.device)
            else:
                raise NotImplementedError

            print("GC the large train data loader")
            del train_loader
            # Force garbage collection
            gc.collect()

        self.time_usage = {
            "latency": 0.0,
            "io_latency": 0.0,
            "compute_latency": 0.0,
            "track_compute": [],  # compute time
            "track_io_model_init": [],  # init model weight
            "track_io_model_load": [],  # load into GPU/CPU
            "track_io_data": [],  # context switch
        }

        # this is to do the expeirment
        self.enable_cache = enable_cache
        if self.enable_cache:
            # todo: warmup for concurrent usage. this is only test for MLP with embedding.
            new_model = self.search_space_ins.new_arch_scratch_with_default_setting("8-8-8-8", bn=False)
            new_model.init_embedding()
            # shared embedding
            manager = Manager()
            self.model_cache = manager.dict()
            self.model_cache["model"] = new_model.embedding
            self.get_cache_data = self._get_cache_data_enabled
            self.set_cache_data = self._set_cache_data_enabled
        else:
            # this is the baseline, independently run
            self.get_cache_data = self._get_cache_data_disabled
            self.set_cache_data = self._set_cache_data_disabled

    def _get_cache_data_enabled(self):
        return self.model_cache["model"]

    def _set_cache_data_enabled(self, data):
        self.model_cache["model"] = data

    def _get_cache_data_disabled(self):
        return None

    def _set_cache_data_disabled(self, data):
        pass

    def if_cuda_avaiable(self):
        if "cuda" in self.device:
            return True
        else:
            return False

    def p1_evaluate(self, data_str: str) -> dict:
        """
        :param data_str: encoded ModelAcquireData
        :return:
        """

        model_acquire = ModelAcquireData.deserialize(data_str)
        return self._p1_evaluate_online(model_acquire)

    def _p1_evaluate_online(self, model_acquire: ModelAcquireData) -> dict:

        model_encoding = model_acquire.model_encoding

        # score using only one metrics
        if self.metrics == CommonVars.PRUNE_SYNFLOW or self.metrics == CommonVars.ExpressFlow:
            bn = False
        else:
            bn = True

        # measure model load time
        begin = time.time()
        new_model = self.search_space_ins.new_arch_scratch_with_default_setting(model_encoding, bn=bn)

        # mlp have embedding layer, which can be cached, optimization!
        if self.search_space_ins.name == Config.MLPSP:
            if self.enable_cache:
                new_model.init_embedding(self.get_cache_data())
                if self.get_cache_data() is None:
                    self.set_cache_data(new_model.embedding.to(self.device))
            else:
                new_model.init_embedding()

        self.time_usage["track_io_model_init"].append(time.time() - begin)

        begin = time.time()
        new_model = new_model.to(self.device)

        self.time_usage["track_io_model_load"].append(time.time() - begin)

        # measure data load time
        begin = time.time()
        mini_batch = self.data_pre_processing(self.metrics, new_model)
        self.time_usage["track_io_data"].append(time.time() - begin)

        _score, curr_time = evaluator_register[self.metrics].evaluate_wrapper(
            arch=new_model,
            device=self.device,
            space_name=self.search_space_ins.name,
            batch_data=mini_batch,
            batch_labels=self.mini_batch_targets)

        self.time_usage["track_compute"].append(curr_time)

        del new_model
        model_score = {self.metrics: _score}
        return model_score

    def data_pre_processing(self, metrics: str, new_model: nn.Module):
        """
        To measure the io/compute time more acccuretely, we pick the data pre_processing here.
        """

        # for those two metrics, we use all one embedding for efficiency (as in their paper)
        if metrics in [CommonVars.ExpressFlow, CommonVars.PRUNE_SYNFLOW]:
            if isinstance(self.mini_batch, torch.Tensor):
                feature_dim = list(self.mini_batch[0, :].shape)
                # add one dimension to feature dim, [1] + [3, 32, 32] = [1, 3, 32, 32]
                mini_batch = torch.ones([1] + feature_dim).float().to(self.device)
            else:
                # this is for the tabular data,
                mini_batch = new_model.generate_all_ones_embedding().float().to(self.device)
        else:
            mini_batch = self.mini_batch

        return mini_batch
