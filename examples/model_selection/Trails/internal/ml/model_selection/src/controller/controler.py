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

from src.controller.core.sample import Sampler
from src.search_space.core.model_params import ModelMicroCfg


class ModelScore:
    def __init__(self, model_id, score):
        self.model_id = model_id
        self.score = score

    def __repr__(self):
        return "m_{}_s_{}".format(self.model_id, self.score)


# for binary insert
def binary_insert_get_rank(rank_list: list, new_item: ModelScore) -> int:
    """
    Insert the new_item to rank_list, then get the rank of it.
    :param rank_list:
    :param new_item:
    :return:
    """
    index = search_position(rank_list, new_item)
    # search the position to insert into
    rank_list.insert(index, new_item)
    return index


# O(logN) search the position to insert into
def search_position(rank_list_m: list, new_item: ModelScore):
    if len(rank_list_m) == 0:
        return 0
    left = 0
    right = len(rank_list_m) - 1
    while left + 1 < right:
        mid = int((left + right) / 2)
        if rank_list_m[mid].score <= new_item.score:
            left = mid
        else:
            right = mid

    # consider the time.
    if rank_list_m[right].score <= new_item.score:
        return right + 1
    elif rank_list_m[left].score <= new_item.score:
        return left + 1
    else:
        return left


class SampleController(object):
    """
    Controller control the sample-score flow in the 1st phase.
    It records the results in the history.
    """

    def __init__(self, search_strategy: Sampler):
        # Current ea is better than others.
        self.search_strategy = search_strategy

        # the large the index, the better the model
        self.ranked_models = []

        # when simple_score_sum=False, records the model's score of each algorithm,
        # use when simple_score_sum=True, record the model's sum score
        self.history = {}

    def sample_next_arch(self) -> (str, ModelMicroCfg):
        """
        Return a generator
        :return:
        """
        return self.search_strategy.sample_next_arch(self.ranked_models)

    def fit_sampler(self, arch_id: str, alg_score: dict, simple_score_sum: bool = False,
                    is_sync: bool = True, arch_micro=None) -> float:
        """
        :param arch_id:
        :param alg_score: {alg_name1: score1, alg_name2: score2}
        :param simple_score_sum: if simply sum multiple scores (good performing),
                             or sum over their rank (worse performing)
        :return:
        """
        if simple_score_sum or len(alg_score.keys()) == 1:
            score = self._use_pure_score_as_final_res(arch_id, alg_score)
        else:
            score = self._use_vote_rank_as_final_res(arch_id, alg_score)
        if is_sync:
            self.search_strategy.fit_sampler(score)
        else:
            self.search_strategy.async_fit_sampler(arch_id, arch_micro, score)
        return score

    def _use_vote_rank_as_final_res(self, model_id: str, alg_score: dict):
        """
        :param model_id:
        :param alg_score: {alg_name1: score1, alg_name2: score2}
        """
        # todo: bug: only all scores' under all arg is greater than previous one, then treat it as greater.
        for alg in alg_score:
            if alg not in self.history:
                self.history[alg] = []

        # add model and score to local list
        for alg, score in alg_score.items():
            binary_insert_get_rank(self.history[alg], ModelScore(model_id, score))

        new_rank_score = self._re_rank_model_id(model_id, alg_score)
        return new_rank_score

    def _use_pure_score_as_final_res(self, model_id: str, alg_score: dict):
        # get the key and sum the score of various alg
        score_sum_key = "_".join(list(alg_score.keys()))
        if score_sum_key not in self.history:
            self.history[score_sum_key] = []
        final_score = 0
        for alg in alg_score:
            final_score += float(alg_score[alg])
        # insert and get rank
        index = binary_insert_get_rank(self.history[score_sum_key], ModelScore(model_id, final_score))
        self.ranked_models.insert(index, model_id)
        return final_score

    def _re_rank_model_id(self, model_id: str, alg_score: dict):
        # todo: re-rank everything, to make it self.ranked_models more accurate.
        model_new_rank_score = {}
        current_explored_models = 0
        for alg, score in alg_score.items():
            for rank_index in range(len(self.history[alg])):
                current_explored_models = len(self.history[alg])
                ms_ins = self.history[alg][rank_index]
                # rank = index + 1, since index can be 0
                if ms_ins.model_id in model_new_rank_score:
                    model_new_rank_score[ms_ins.model_id] += rank_index + 1
                else:
                    model_new_rank_score[ms_ins.model_id] = rank_index + 1

        for ele in model_new_rank_score.keys():
            model_new_rank_score[ele] = model_new_rank_score[ele] / current_explored_models

        self.ranked_models = [k for k, v in sorted(model_new_rank_score.items(), key=lambda item: item[1])]
        new_rank_score = model_new_rank_score[model_id]
        return new_rank_score

    def get_current_top_k_models(self, k=-1):
        """
        The model is already scored by: low -> high
        :param k:
        :return:
        """
        if k == -1:
            # retur all models
            return self.ranked_models
        else:
            return self.ranked_models[-k:]


if __name__ == "__main__":

    rank_list = []
    begin = time.time()
    score_list = [1, 2, 3, 1, 2]
    for i in range(5):
        ms = ModelScore(i, score_list[i])
        binary_insert_get_rank(rank_list, ms)
    print(rank_list)
    print(time.time() - begin)

    rank_list = []
    begin = time.time()
    score_list = [1, 1, 1, 1, 1]
    for i in range(5):
        ms = ModelScore(i, score_list[i])
        binary_insert_get_rank(rank_list, ms)
    print(rank_list)
    print(time.time() - begin)
