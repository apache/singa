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

from src.common.constant import Config
from src.eva_engine.phase2.algo.trainer import ModelTrainer
from src.logger import logger
from src.query_api.interface import SimulateTrain
from src.search_space.core.space import SpaceWrapper
from torch.utils.data import DataLoader


class P2Evaluator:

    def __init__(self,
                 search_space_ins: SpaceWrapper,
                 dataset: str,
                 is_simulate: bool = True,
                 train_loader: DataLoader = None,
                 val_loader: DataLoader = None,
                 args=None):
        """
        :param search_space_ins:
        :param dataset:
        :param is_simulate: train or not, default query from API.
        """
        self.search_space_ins = search_space_ins

        # dataset name
        self.dataset = dataset
        self.is_simulate = is_simulate
        self.acc_getter = None

        # for training only
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args

    def p2_evaluate(self, cand: str, epoch_per_model: int) -> (float, float):
        """
        :param cand: candidate id
        :param epoch_per_model: epoch for each model
        :return:
        """
        # if it's simulate or it's image dataset
        if self.is_simulate or self.search_space_ins.name in [Config.NB101, Config.NB201]:
            return self._evaluate_query(cand, epoch_per_model)
        else:
            return self._evaluate_train(cand, epoch_per_model)

    def _evaluate_query(self, cand: str, epoch_per_model: int) -> (float, float):
        """
        :param cand: the candidate to evaluate
        :param epoch_per_model: how many resource it can use, epoch number
        :return:
        """
        if self.acc_getter is None:
            self.acc_getter = SimulateTrain(space_name=self.search_space_ins.name)

        acc, time_usage = self.acc_getter.get_ground_truth(arch_id=cand, epoch_num=epoch_per_model, dataset=self.dataset)

        return acc, time_usage

    def _evaluate_train(self, cand: str, epoch_per_model: int) -> (float, float):
        """
        :param cand: the candidate to evaluate
        :param epoch_per_model: how many resource it can use, epoch number
        :return:
        """
        model = self.search_space_ins.new_architecture(cand)
        valid_auc, total_run_time, train_log = ModelTrainer.fully_train_arch(
           model=model,
           use_test_acc=False,
           epoch_num=epoch_per_model,
           train_loader=self.train_loader,
           val_loader=self.val_loader,
           test_loader=self.val_loader,
           args=self.args)

        logger.info(f' ----- model id: {cand}, Val_AUC : {valid_auc} Total running time: '
                    f'{total_run_time}-----')

        return valid_auc, total_run_time
