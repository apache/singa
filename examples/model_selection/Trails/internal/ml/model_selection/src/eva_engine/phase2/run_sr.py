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
from src.common.constant import Config
from src.eva_engine.phase2.evaluator import P2Evaluator
from src.search_space.core.space import SpaceWrapper


class BudgetAwareControllerSR:

    @staticmethod
    def pre_calculate_epoch_required(K, U, eta: int = 3, max_unit_per_model: int = 200):
        """
        :param K: candidates lists
        :param U: min resource each candidate needs
        :return:
        """
        total_epoch_each_rounds = K * U
        min_budget_required = 0

        previous_epoch = None
        while True:
            cur_cand_num = K
            if cur_cand_num == 1:
                break
            # number of each res given to each cand, pick lower bound
            epoch_per_model = int(total_epoch_each_rounds / cur_cand_num)
            if previous_epoch is None:
                previous_epoch = epoch_per_model
            elif previous_epoch == epoch_per_model:
                # current epoch  == last epoch, no need to re-evaluate each component
                K = cur_cand_num - 1
                continue

            previous_epoch = epoch_per_model

            if epoch_per_model >= max_unit_per_model:
                epoch_per_model = max_unit_per_model

            # print(f"[successive_reject]: {cur_cand_num} model left, "
            #       f"and evaluate each model with {epoch_per_model} epoch, total epoch = {max_unit_per_model}")
            # evaluate each arch
            min_budget_required += epoch_per_model * cur_cand_num
            # sort from min to max
            if epoch_per_model == max_unit_per_model:
                # each model is fully evaluated, just return top 1
                K = 1
            else:
                # only keep 1/eta, pick lower bound
                K = cur_cand_num - 1
        return min_budget_required

    def __init__(self,
                 search_space_ins: SpaceWrapper, dataset_name: str,
                 eta, args, time_per_epoch):

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
        self.name = "SUCCREJCT"

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
        if len(history) == 0:
            raise f"{fixed_time_budget} is too small for current config"
        return history[-1]

    def run_phase2(self, U: int, candidates_m: list):
        """
        :param candidates_m: candidates lists
        :param U: min resource each candidate needs
        :return:
        """
        total_time = 0
        # print(f" *********** begin BudgetAwareControllerSR with U={U}, K={len(candidates_m)} ***********")
        candidates = copy(candidates_m)
        total_epoch_each_rounds = len(candidates) * U
        min_budget_required = 0
        previous_epoch = None
        scored_cand = None
        while True:
            cur_cand_num = len(candidates)
            if cur_cand_num == 1:
                break
            total_score = []
            # number of each res given to each cand, pick lower bound
            epoch_per_model = int(total_epoch_each_rounds / cur_cand_num)

            if previous_epoch is None:
                previous_epoch = epoch_per_model
            elif previous_epoch == epoch_per_model:
                # which means the epoch don't increase, no need to re-evaluate each component
                num_keep = cur_cand_num - 1
                candidates = [ele[0] for ele in scored_cand[-num_keep:]]
                continue

            previous_epoch = epoch_per_model

            if epoch_per_model >= self.max_unit_per_model:
                epoch_per_model = self.max_unit_per_model

            # print(f"[successive_reject]: {cur_cand_num} model left, "
            #       f"and evaluate each model with {epoch_per_model} epoch, total epoch = {self.max_unit_per_model}")
            # evaluate each arch
            for cand in candidates:
                score, time_usage = self._evaluator.p2_evaluate(cand, epoch_per_model)
                total_time += time_usage
                total_score.append((cand, score))
                min_budget_required += epoch_per_model
            # sort from min to max
            scored_cand = sorted(total_score, key=lambda x: x[1])

            if epoch_per_model == self.max_unit_per_model:
                # each model is fully evaluated, just return top 1
                candidates = [scored_cand[-1][0]]
            else:
                # only keep m-1, remove the worst one
                num_keep = cur_cand_num - 1
                candidates = [ele[0] for ele in scored_cand[-num_keep:]]

        best_perform, _ = self._evaluator.p2_evaluate(candidates[0], self.max_unit_per_model)
        return candidates[0], best_perform, min_budget_required, total_time
