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
from src.eva_engine.phase2.run_sh import BudgetAwareControllerSH
from src.logger import logger
from src.search_space.core.space import SpaceWrapper

eta = 3


def min_budget_calculation(search_space_ins: SpaceWrapper, dataset: str,
                           N_K_ratio: int, sh: BudgetAwareControllerSH, t1_: float):
    # Calculate the minimum budget requirements for both phases
    K_max = int(len(search_space_ins) / N_K_ratio)

    if search_space_ins.name == Config.NB101:
        U_options = [4, 12, 16, 108]
    elif search_space_ins.name == Config.NB201:
        U_options = list(range(1, 200))
    elif search_space_ins.name == Config.MLPSP:
        # TODO: This is for benchmark only
        if dataset == Config.Frappe:
            MaxEpochTrained = 20
        elif dataset == Config.Criteo:
            MaxEpochTrained = 10
        elif dataset == Config.UCIDataset:
            MaxEpochTrained = 40
        else:
            raise NotImplementedError
        U_options = list(range(1, MaxEpochTrained))
    else:
        raise NotImplementedError

    U_min = U_options[0]
    min_budget_required_both_phase = sh.pre_calculate_time_required(K=1, U=U_min)[1] + N_K_ratio * t1_

    return K_max, U_options, U_min, min_budget_required_both_phase


def schedule(dataset: str, sh: BudgetAwareControllerSH, T_: float, t1_: float, t2_: float, w_: int,
             search_space_ins: SpaceWrapper, N_K_ratio: int,
             only_phase1: bool = False):
    """
    :param dataset
    :param sh: BudgetAwareControllerSH instnace
    :param T_: user given time budget
    :param t1_: time to score one model
    :param t2_: time to train one model
    :param w_: number of workers, for parallelly running.
    :param search_space_ins: search spcae instance
    :param N_K_ratio: N/K = N_K_ratio
    :param only_phase1: Only use filtering phase.
    """
    if T_ < 1:
        raise ValueError('Total time budget must be greater than 1 second')

    K_max, U_options, U_min, min_budget_required_both_phase = min_budget_calculation(
        search_space_ins, dataset, N_K_ratio, sh, t1_)

    # collection of (best_K, best_U, best_N)
    history = []

    # Calculate phase 1
    time_used = t1_
    enable_phase2_at_least = sh.pre_calculate_time_required(K=2, U=U_min)[1] + 2 * N_K_ratio * t1_

    if only_phase1 or enable_phase2_at_least > T_:
        # all time give to phase1, explore n models
        N_only = min(int(T_ / t1_), len(search_space_ins))
        history.extend([(1, U_min, i) for i in range(1, N_only + 1) if i * t1_ <= T_])
        if not history:
            raise ValueError(
                f' [trails] Only p1, Budget {T_} is too small, it\'s at least >= {time_used} with current worker, '
                f'{t1_}, {t2_}, eta')

    # Calculate phase 2, start from min U, if user given budget is larger enough, then evaluat each mode with more epoch
    else:
        # record all possible K, U pair meeting the SLO ( time used < T)
        for K_ in range(2, min(int(T_ / t1_), K_max) + 1):
            N_ = K_ * N_K_ratio
            for U in U_options:
                time_used = sh.pre_calculate_time_required(K=K_, U=U)[1] + N_ * t1_
                if time_used > T_:
                    break
                else:
                    history.append((K_, U, N_))
        if not history:
            raise ValueError(
                f' [trails] Budget {T_} is too small, it\'s at least >= {min_budget_required_both_phase}'
                f' with current worker, {t1_}, {t2_}, eta')

    best_K, best_U, best_N = history[-1]
    N_scored = best_N
    B1_time_used = N_scored * t1_
    B2_all_epoch, B2_time_used = sh.pre_calculate_time_required(K=best_K, U=best_U)

    logger.info(
        f' [trails] The schedule result: when T = {T_} second, N = {N_scored}, K = {best_K}, best_U = {best_U}, '
        f'time_used = {B1_time_used + B2_time_used}')
    return best_K, best_U, N_scored, B1_time_used, B2_time_used, B2_all_epoch
