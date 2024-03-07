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

from src.common.constant import Config, CommonVars
from src.query_api.query_api_img import Gt201, Gt101
from src.query_api.query_api_mlp import GTMLP
from src.query_api.query_api_img import ImgScoreQueryApi
from typing import *


def profile_NK_trade_off(dataset):
    """
    This is get from the profling result.  
    We try various N/K combinations, and find this is better.
    """
    if dataset == Config.c10:
        return 100
    elif dataset == Config.c100:
        return 100
    elif dataset == Config.imgNet:
        return 100
    else:
        # this is the expressflow
        # return 30
        # this is the jacflow
        return 100


class SimulateTrain:

    def __init__(self, space_name: str):
        """
        :param space_name: NB101 or NB201, MLP
        """
        self.space_name = space_name
        self.api = None

    # get the test_acc and time usage to train of this arch_id
    def get_ground_truth(self, arch_id: str, dataset: str, epoch_num: int = None, total_epoch: int = 200):
        """
        :param arch_id: 
        :param dataset: 
        :param epoch_num: which epoch's performance to return
        :param total_epoch: 
        """
        if self.space_name == Config.NB101:
            self.api = Gt101()
            acc, time_usage = self.api.get_c10_test_info(arch_id, dataset, epoch_num)
            return acc, time_usage

        elif self.space_name == Config.NB201:
            self.api = Gt201()
            if total_epoch == 200:
                acc, time_usage = self.api.query_200_epoch(arch_id, dataset, epoch_num)
            else:  # 12
                acc, time_usage = self.api.query_12_epoch(arch_id, dataset, epoch_num)
            return acc, time_usage

        elif self.space_name == Config.MLPSP:
            self.api = GTMLP(dataset)
            acc, time_usage = self.api.get_valid_auc(arch_id, epoch_num)
            return acc, time_usage

        else:
            raise NotImplementedError

    # get the high acc of k arch with highest score
    def get_high_acc_top_10(self, top10):
        all_top10_acc = []
        time_usage = 0
        for arch_id in top10:
            score_, time_usage_ = self.get_ground_truth(arch_id)
            all_top10_acc.append(score_)
            time_usage += time_usage_
        return max(all_top10_acc), time_usage

    def get_best_arch_id(self, top10):
        cur_best = 0
        res = None
        for arch_id in top10:
            acc, _ = self.get_ground_truth(arch_id)
            if acc > cur_best:
                cur_best = acc
                res = arch_id
        return res

    def query_all_model_ids(self, dataset):
        if self.space_name == Config.NB101:
            self.api = Gt101()
        elif self.space_name == Config.NB201:
            self.api = Gt201()
        elif self.space_name == Config.MLPSP:
            self.api = GTMLP(dataset)
        return self.api.get_all_trained_model_ids()


class SimulateScore:
    def __init__(self, space_name: str, dataset_name: str):
        """
        :param space_name: NB101 or NB201, MLP
        :param dataset_name: NB101 or NB201, MLP
        """
        self.space_name = space_name
        if self.space_name == Config.MLPSP:
            self.api = GTMLP(dataset_name)
        else:
            self.api = ImgScoreQueryApi(self.space_name, dataset_name)

    # get the test_acc and time usage to train of this arch_id
    def query_tfmem_rank_score(self, arch_id) -> Dict:
        # todo: here we use the global rank, other than dymalically update the rank
        # todo: so, we directly return the rank_score, instead of the mutilpel_algs score
        # return {"nas_wot": self.api.get_metrics_score(arch_id, dataset)["nas_wot"],
        #         "synflow": self.api.get_metrics_score(arch_id, dataset)["synflow"],
        #         }
        return self.api.get_global_rank_score(arch_id)

    def query_all_tfmem_score(self, arch_id) -> Dict:
        """
        return {alg_name: score}
        """
        return self.api.api_get_score(arch_id)

    def query_all_model_ids(self, dataset) -> List:
        """
        return all models_ids as a list
        """
        return self.api.get_all_scored_model_ids()
