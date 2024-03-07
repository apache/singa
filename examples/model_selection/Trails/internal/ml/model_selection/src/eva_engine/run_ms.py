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


import time

from typing import Set, List

from src.eva_engine import coordinator
from src.eva_engine.phase1.run_phase1 import RunPhase1, p1_evaluate_query
from torch.utils.data import DataLoader
from src.eva_engine.phase2.run_sh import BudgetAwareControllerSH
from src.eva_engine.phase2.run_sr import BudgetAwareControllerSR
from src.eva_engine.phase2.run_uniform import UniformAllocation
from src.logger import logger
from src.search_space.init_search_space import init_search_space
from src.query_api.interface import profile_NK_trade_off
from src.common.constant import Config


class RunModelSelection:

    def __init__(self, search_space_name: str, args, is_simulate: bool = False):
        self.args = args

        self.eta = 3
        self.is_simulate = is_simulate
        # basic
        self.search_space_name = search_space_name
        self.dataset = self.args.dataset

        # p2 evaluator
        self.sh = None

        # instance of the search space.
        self.search_space_ins = init_search_space(self.args)

    def select_model_simulate(self, budget: float, run_id: int = 0, only_phase1: bool = False, run_workers: int = 1):
        """
        This is for image data only
        """

        # 0. profiling dataset and search space, get t1 and t2

        score_time_per_model, train_time_per_epoch, N_K_ratio = self.search_space_ins.profiling(self.dataset)
        self.sh = BudgetAwareControllerSH(
            search_space_ins=self.search_space_ins,
            dataset_name=self.dataset,
            eta=self.eta,
            time_per_epoch=train_time_per_epoch,
            args=self.args,
            is_simulate=self.is_simulate)

        # 1. run coordinator to schedule
        K, U, N, B1_planed_time, B2_planed_time, B2_all_epoch = coordinator.schedule(self.dataset, self.sh, budget,
                                                                                     score_time_per_model,
                                                                                     train_time_per_epoch,
                                                                                     run_workers,
                                                                                     self.search_space_ins,
                                                                                     N_K_ratio,
                                                                                     only_phase1)

        print(f"Budget = {budget}, N={N}, K={K}")

        # 2. run phase 1 to score N models
        k_models, B1_actual_time_use = p1_evaluate_query(self.search_space_name, self.dataset, run_id, N, K)

        # 3. run phase-2 to determine the final model
        best_arch, best_arch_performance, B2_actual_epoch_use, _ = self.sh.run_phase2(U, k_models)
        # print("best model returned from Phase2 = ", k_models)

        return best_arch, B1_actual_time_use + B2_actual_epoch_use * train_time_per_epoch, \
               B1_planed_time + B2_planed_time, B2_all_epoch

    def select_model_online_clean(self, budget: float, data_loader: List[DataLoader],
                                  only_phase1: bool = False, run_workers: int = 1):
        """
        Select model online for structured data.
        :param budget:  time budget
        :param data_loader:  time budget
        :param only_phase1:
        :param run_workers:
        :return:
        """
        begin_time = time.time()
        logger.info("1. profiling....")
        score_time_per_model = self.profile_filtering(data_loader)
        train_time_per_epoch = self.profile_refinement(data_loader)
        logger.info("2. coordination....")
        K, U, N = self.coordination(budget, score_time_per_model, train_time_per_epoch, only_phase1)
        logger.info("3. filtering phase....")
        k_models, all_models, p1_trace_highest_score, p1_trace_highest_scored_models_id = self.filtering_phase(
            N, K, train_loader=data_loader[0])
        logger.info("4. refinement phase....")
        best_arch, best_arch_performance, _, _ = self.refinement_phase(
            U, k_models, train_loader=data_loader[0], valid_loader=data_loader[1])

        end_time = time.time()
        real_time_usage = end_time - begin_time

        return best_arch, best_arch_performance, real_time_usage, all_models, \
               p1_trace_highest_score, p1_trace_highest_scored_models_id

    def select_model_online(self, budget: float, data_loader: List[DataLoader],
                            only_phase1: bool = False, run_workers: int = 1):
        """
        Select model online for structured data.
        :param budget:  time budget
        :param data_loader:  time budget
        :param only_phase1:
        :param run_workers:
        :return:
        """

        train_loader, valid_loader, test_loader = data_loader

        logger.info(f"0. [trails] Begin model selection, is_simulate={self.is_simulate} ... ")
        begin_time = time.time()

        logger.info("1. [trails] Begin profiling.")
        # 0. profiling dataset and search space, get t1 and t2
        score_time_per_model, train_time_per_epoch, N_K_ratio = self.search_space_ins.profiling(
            self.dataset,
            train_loader,
            valid_loader,
            self.args,
            is_simulate=self.is_simulate)

        self.sh = BudgetAwareControllerSH(
            search_space_ins=self.search_space_ins,
            dataset_name=self.dataset,
            eta=self.eta,
            time_per_epoch=train_time_per_epoch,
            is_simulate=self.is_simulate,
            train_loader=train_loader,
            val_loader=valid_loader,
            args=self.args)

        # 1. run coordinator to schedule
        logger.info("2. [trails] Begin scheduling...")
        K, U, N, B1_planed_time, B2_planed_time, B2_all_epoch = coordinator.schedule(self.dataset, self.sh, budget,
                                                                                     score_time_per_model,
                                                                                     train_time_per_epoch,
                                                                                     run_workers,
                                                                                     self.search_space_ins,
                                                                                     N_K_ratio,
                                                                                     only_phase1)

        print(f"Budget = {budget}, N={N}, K={K}")

        # 2. run phase 1 to score N models
        logger.info("3. [trails] Begin to run phase1: filter phase")
        # lazy loading the search space if needed.

        # run phase-1 to get the K models.
        p1_runner = RunPhase1(
            args=self.args,
            K=K, N=N,
            search_space_ins=self.search_space_ins,
            train_loader=train_loader,
            is_simulate=self.is_simulate)

        k_models, all_models, p1_trace_highest_score, p1_trace_highest_scored_models_id \
            = p1_runner.run_phase1()

        logger.info("4. [trails] Begin to run phase2: refinement phase")

        # 3. run phase-2 to determine the final model
        best_arch, best_arch_performance, B2_actual_epoch_use, _ = self.sh.run_phase2(U, k_models)
        # print("best model returned from Phase2 = ", k_models)
        end_time = time.time()
        real_time_usage = end_time - begin_time
        planned_time_usage = B1_planed_time + B2_planed_time
        logger.info("5.  [trails] Real time Usage = " + str(real_time_usage)
                    + ", Final selected model = " + str(best_arch)
                    + ", planned time usage = " + str(planned_time_usage)
                    )
        # best arch returned,
        # time usage, epoch trained,
        # p1 ea trace
        return best_arch, best_arch_performance, \
               real_time_usage, planned_time_usage, B2_all_epoch, \
               all_models, p1_trace_highest_score, p1_trace_highest_scored_models_id

    def schedule_only(self, budget: float, data_loader: List[DataLoader],
                      only_phase1: bool = False, run_workers: int = 1):
        """
        Select model online
        :param budget:  time budget
        :param data_loader:  time budget
        :param only_phase1:
        :param run_workers:
        :return:
        """

        train_loader, valid_loader, test_loader = data_loader

        logger.info("0. [trails] Begin model selection ... ")

        logger.info("1. [trails] Begin profiling.")
        # 0. profiling dataset and search space, get t1 and t2
        score_time_per_model, train_time_per_epoch, N_K_ratio = self.search_space_ins.profiling(
            self.dataset,
            train_loader,
            valid_loader,
            self.args,
            is_simulate=self.is_simulate)

        self.sh = BudgetAwareControllerSH(
            search_space_ins=self.search_space_ins,
            dataset_name=self.dataset,
            eta=self.eta,
            time_per_epoch=train_time_per_epoch,
            is_simulate=self.is_simulate,
            train_loader=train_loader,
            val_loader=valid_loader,
            args=self.args)

        # 1. run coordinator to schedule
        logger.info("2. [trails] Begin scheduling...")
        K, U, N, B1_planed_time, B2_planed_time, B2_all_epoch = coordinator.schedule(self.dataset, self.sh, budget,
                                                                                     score_time_per_model,
                                                                                     train_time_per_epoch,
                                                                                     run_workers,
                                                                                     self.search_space_ins,
                                                                                     N_K_ratio,
                                                                                     only_phase1)

        return K, U, N, B1_planed_time, B2_planed_time, B2_all_epoch

    #############################################
    # to support in-database model selection
    #############################################

    def profile_filtering(self, data_loader: List[DataLoader] = [None, None, None]):
        logger.info("0. [trails] Begin profile_filtering...")
        begin_time = time.time()
        train_loader, valid_loader, test_loader = data_loader
        score_time_per_model = self.search_space_ins.profiling_score_time(
            self.dataset,
            train_loader,
            valid_loader,
            self.args,
            is_simulate=self.is_simulate)
        logger.info(f"0. [trails] profile_filtering Done, time_usage = {time.time() - begin_time}")
        return score_time_per_model

    def profile_refinement(self, data_loader: List[DataLoader] = [None, None, None]):
        logger.info("0. [trails] Begin profile_refinement...")
        begin_time = time.time()
        train_loader, valid_loader, test_loader = data_loader
        train_time_per_epoch = self.search_space_ins.profiling_train_time(
            self.dataset,
            train_loader,
            valid_loader,
            self.args,
            is_simulate=self.is_simulate)
        logger.info(f"0. [trails] profile_refinement Done, time_usage = {time.time() - begin_time}")
        return train_time_per_epoch

    def coordination(self, budget: float, score_time_per_model: float, train_time_per_epoch: float, only_phase1: bool):
        logger.info("1. [trails] Begin coordination...")
        begin_time = time.time()
        sh = BudgetAwareControllerSH(
            search_space_ins=self.search_space_ins,
            dataset_name=self.dataset,
            eta=self.eta,
            time_per_epoch=train_time_per_epoch,
            is_simulate=self.is_simulate,
            train_loader=None,
            val_loader=None,
            args=self.args)
        n_k_ratio = profile_NK_trade_off(self.dataset)
        K, U, N, B1_planed_time, B2_planed_time, B2_all_epoch = coordinator.schedule(
            self.dataset, sh, budget,
            score_time_per_model,
            train_time_per_epoch,
            1,
            self.search_space_ins,
            n_k_ratio,
            only_phase1)

        logger.info(f"1. [trails] Coordination Done, time_usage = {time.time() - begin_time}")
        return K, U, N

    def filtering_phase(self, N, K, train_loader=None):
        logger.info("2. [trails] Begin filtering_phase...")
        begin_time = time.time()
        p1_runner = RunPhase1(
            args=self.args,
            K=K, N=N,
            search_space_ins=self.search_space_ins,
            train_loader=train_loader,
            is_simulate=self.is_simulate)

        k_models, all_models, p1_trace_highest_score, p1_trace_highest_scored_models_id \
            = p1_runner.run_phase1()
        logger.info(f"2. [trails] filtering_phase Done, time_usage = {time.time() - begin_time}")
        print(f"2. [trails] filtering_phase Done, time_usage = {time.time() - begin_time}")
        return k_models, all_models, p1_trace_highest_score, p1_trace_highest_scored_models_id

    def refinement_phase(self, U, k_models, alg_name: str = Config.SUCCHALF, train_loader=None, valid_loader=None,
                         train_time_per_epoch=None):
        logger.info("3. [trails] Begin refinement...")
        begin_time = time.time()

        if alg_name == Config.SUCCHALF:
            self.sh = BudgetAwareControllerSH(
                search_space_ins=self.search_space_ins,
                dataset_name=self.dataset,
                eta=self.eta,
                time_per_epoch=train_time_per_epoch,
                is_simulate=self.is_simulate,
                train_loader=train_loader,
                val_loader=valid_loader,
                args=self.args)
        elif alg_name == Config.SUCCREJCT:
            self.sh = BudgetAwareControllerSR(
                search_space_ins=self.search_space_ins,
                dataset_name=self.dataset,
                eta=self.eta,
                time_per_epoch=train_time_per_epoch,
                args=self.args)
        elif alg_name == Config.UNIFORM:
            self.sh = UniformAllocation(
                search_space_ins=self.search_space_ins,
                dataset_name=self.dataset,
                eta=self.eta,
                time_per_epoch=train_time_per_epoch,
                args=self.args)
        else:
            raise NotImplementedError

        best_arch, best_arch_performance, B2_actual_epoch_use, total_time_usage = self.sh.run_phase2(U, k_models)
        logger.info(
            f"3. [trails] refinement phase Done, time_usage = {time.time() - begin_time}, "
            f"epoches_used = {B2_actual_epoch_use}")
        return best_arch, best_arch_performance, B2_actual_epoch_use, total_time_usage
