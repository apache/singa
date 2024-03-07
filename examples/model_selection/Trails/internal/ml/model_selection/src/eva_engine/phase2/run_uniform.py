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


from copy import copy
from src.search_space.core.space import SpaceWrapper
from src.common.constant import Config
from src.eva_engine.phase2.evaluator import P2Evaluator


# UniformAllocation
class UniformAllocation:

    def __init__(self,
                 search_space_ins: SpaceWrapper, dataset_name: str,
                 eta, time_per_epoch, args=None):

        self.is_simulate = True
        self._evaluator = P2Evaluator(search_space_ins,
                                      dataset_name,
                                      is_simulate=True,
                                      train_loader=None,
                                      val_loader=None,
                                      args=None)
        self.eta = eta
        self.max_unit_per_model = args.epoch
        self.time_per_epoch = time_per_epoch
        self.name = "UNIFORM"

    def schedule_budget_per_model_based_on_T(self, space_name, fixed_time_budget, K_):
        # for benchmarking only phase 2

        # try different K and U combinations
        # only consider 15625 arches in this paper
        # min_budget_required: when K = 1, N = min_budget_required * 1
        if space_name == Config.NB101:
            U_options = [4, 12, 16, 108]
        else:
            U_options = list(range(1, 200))

        history = []

        for U in U_options:
            expected_time_used = self.pre_calculate_epoch_required(K_, U) * self.time_per_epoch
            if expected_time_used > fixed_time_budget:
                break
            else:
                history.append(U)
        return history[-1]

    def pre_calculate_epoch_required(self, K, U, eta: int=3, max_unit_per_model: int=200):
        """
        :param B: total budget for phase 2
        :param U: mini unit computation for each modle
        :param candidates_m:
        :return:
        """
        return K * U

    def run_phase2(self, U: int, candidates_m: list):
        """
        :param U: mini unit computation for each modle
        :param candidates_m:
        :return:
        """

        # print(f" *********** begin uniformly_allocate with U={U}, K={len(candidates_m)} ***********")
        candidates = copy(candidates_m)
        min_budget_required = 0

        # todo: this is to run the full training, when compute full traiing
        # U = self.max_unit_per_model

        if U >= self.max_unit_per_model:
            U = self.max_unit_per_model

        # print(f"[uniformly_allocate]: uniformly allocate {U} epoch to each model")

        total_time = 0
        total_score = []
        for cand in candidates:
            score, time_usage = self._evaluator.p2_evaluate(cand, U)
            total_time += time_usage
            total_score.append((cand, score))
            min_budget_required += U
        # sort from min to max
        scored_cand = sorted(total_score, key=lambda x: x[1])
        candidate = scored_cand[-1][0]
        best_perform, _ = self._evaluator.p2_evaluate(candidate, self.max_unit_per_model)
        return candidate, best_perform, min_budget_required, total_time
