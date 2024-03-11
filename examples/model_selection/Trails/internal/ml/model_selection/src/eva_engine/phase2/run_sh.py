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

# successive halving
from src.logger import logger
from src.search_space.core.space import SpaceWrapper
from torch.utils.data import DataLoader


class BudgetAwareControllerSH:

    @staticmethod
    def pre_calculate_epoch_required(K: int, U: int, eta: int=3, max_unit_per_model: int=200):
        if K == 1:
            return 0

        cur_cand_num = K
        cur_epoch = min(U, max_unit_per_model)  # Limit the current epoch to max_unit_per_model
        total_epochs = 0

        while cur_cand_num > 1 and cur_epoch < max_unit_per_model:
            total_epochs += cur_cand_num * cur_epoch
            # Prune models
            cur_cand_num = int(cur_cand_num * (1 / eta))
            # Increase the training epoch for the remaining models
            cur_epoch = min(cur_epoch * eta, max_unit_per_model)

        # If the models are fully trained and there is more than one candidate, add these final evaluations to the total
        if cur_cand_num > 1 and cur_epoch >= max_unit_per_model:
            total_epochs += cur_cand_num * max_unit_per_model

        return total_epochs

    def __init__(self,
                 search_space_ins: SpaceWrapper, dataset_name: str,
                 eta, time_per_epoch,
                 train_loader: DataLoader = None,
                 val_loader: DataLoader = None,
                 args=None,
                 is_simulate: bool = True):
        """
        :param search_space_ins:
        :param dataset_name:
        :param time_per_epoch:
        :param is_simulate:
        :param eta: 1/mu to keep in each iteration
        """
        self.is_simulate = is_simulate
        self._evaluator = P2Evaluator(search_space_ins, dataset_name,
                                      is_simulate=is_simulate,
                                      train_loader=train_loader, val_loader=val_loader,
                                      args=args)
        self.eta = eta
        self.max_unit_per_model = args.epoch
        self.time_per_epoch = time_per_epoch
        self.name = "SUCCHALF"

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
            real_time_used = \
                BudgetAwareControllerSH.pre_calculate_epoch_required(
                    self.eta, self.max_unit_per_model, K_, U) * self.time_per_epoch

            if real_time_used > fixed_time_budget:
                break
            else:
                history.append(U)
        if len(history) == 0:
            print(f"{fixed_time_budget} is too small for current config")
            raise f"{fixed_time_budget} is too small for current config"
        return history[-1]

    def pre_calculate_time_required(self, K, U):
        all_epoch = BudgetAwareControllerSH.pre_calculate_epoch_required(self.eta, self.max_unit_per_model, K, U)
        return all_epoch, all_epoch * self.time_per_epoch

    def run_phase2(self, U: int, candidates_m: list) -> (str, float, float):
        total_time = 0
        if len(candidates_m) == 0:
            raise "No model to explore during the second phase!"
        candidates_m_ori = copy(candidates_m)
        if len(candidates_m) == 1:
            best_perform, _ = self._evaluator.p2_evaluate(candidates_m[0], self.max_unit_per_model)
            return candidates_m[0], best_perform, 0, 0

        eta = self.eta
        max_unit_per_model = self.max_unit_per_model

        cur_cand_num = len(candidates_m)
        cur_epoch = min(U, max_unit_per_model)  # Limit the current epoch to max_unit_per_model
        total_epochs = 0

        while cur_cand_num > 1 and cur_epoch < max_unit_per_model:
            logger.info(f"4. [trails] Running phase2: train {len(candidates_m)} models each with {cur_epoch} epochs")
            scores = []
            # Evaluate all models
            for cand in candidates_m:
                score, time_usage = self._evaluator.p2_evaluate(cand, cur_epoch)
                scores.append((score, cand))
                total_epochs += cur_epoch
                total_time += time_usage

            # Sort models based on score
            scores.sort(reverse=True, key=lambda x: x[0])

            # Prune models, at lease keep one model
            cur_cand_num = max(int(cur_cand_num * (1 / eta)), 1)
            candidates_m = [x[1] for x in scores[:cur_cand_num]]

            # Increase the training epoch for the remaining models
            cur_epoch = min(cur_epoch * eta, max_unit_per_model)

        # If the models can be fully trained and there is more than one candidate, select the top one
        if cur_cand_num > 1 and cur_epoch >= max_unit_per_model:
            logger.info(
                f"4. [trails] Running phase2: train {len(candidates_m)} models each with {max_unit_per_model} epochs")
            scores = []
            for cand in candidates_m:
                score, time_usage = self._evaluator.p2_evaluate(cand, max_unit_per_model)
                scores.append((score, cand))
                total_epochs += cur_epoch
                total_time += time_usage
            scores.sort(reverse=True, key=lambda x: x[0])
            candidates_m = [scores[0][1]]

        # only return the performance when simulating, skip the training, just return model
        if self.is_simulate:
            logger.info(
                f"5. [trails] Phase2 Done, Select {candidates_m[0]}, "
                f"simulate={self.is_simulate}. Acqure the ground truth")
            best_perform, _ = self._evaluator.p2_evaluate(candidates_m[0], self.max_unit_per_model)
        else:
            logger.info(
                f"5. [trails] Phase2 Done, Select {candidates_m[0]}, "
                f"simulate={self.is_simulate}, Skip training")
            best_perform = 0
        # Return the best model and the total epochs used
        return candidates_m[0], best_perform, total_epochs, total_time


if __name__ == "__main__":
    'frappe: 20, uci_diabetes: 40, criteo: 10'
    'nb101: 108, nb201: 200'
    k_options = [1, 2, 4, 8, 16]
    u_options = [1, 2, 4, 8, 16]
    print(f"k={10}, u={8}, total_epoch = {BudgetAwareControllerSH.pre_calculate_epoch_required(3, 20, 10, 8)}")
    for k in k_options:
        for u in u_options:
            print(f"k={k}, u={u}, total_epoch = {BudgetAwareControllerSH.pre_calculate_epoch_required(3, 20, k, u)}")
