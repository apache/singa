

from copy import copy
from random import randint
from src.common.constant import Config


# UniformAllocation
class UniformAllocation:

    def __init__(self, evaluator, time_per_epoch, max_unit=200):
        """
        :param evaluator:
        :param max_unit:  for 201, it's 200, for 101 it's 108
        """
        self._evaluator = evaluator
        self.max_unit_per_model = max_unit
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

    def pre_calculate_epoch_required(self, K, U):
        """
        :param B: total budget for phase 2
        :param U: mini unit computation for each modle
        :param candidates_m:
        :return:
        """
        return K*U

    def run_phase2(self, U: int, candidates_m: list):
        """
        :param U: mini unit computation for each modle
        :param candidates_m:
        :return:
        """

        # print(f" *********** begin uniformly_allocate with U={U}, K={len(candidates_m)} ***********")

        candidates = copy(candidates_m)
        min_budget_required = 0

        if U >= self.max_unit_per_model:
            U = self.max_unit_per_model

        # print(f"[uniformly_allocate]: uniformly allocate {U} epoch to each model")

        total_score = []
        for cand in candidates:
            score = self._evaluator.p2_evaluate(cand, U)
            total_score.append((cand, score))
            min_budget_required += U
        # sort from min to max
        scored_cand = sorted(total_score, key=lambda x: x[1])
        candidate = scored_cand[-1][0]
        return candidate, None, min_budget_required


