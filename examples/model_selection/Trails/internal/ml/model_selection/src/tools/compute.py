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


# for binary insert
from typing import List
import numpy as np


def binary_insert_get_rank(rank_list: list, new_item: List) -> int:
    """
    Insert the new_item to rank_list, then get the rank of it.
    :param rank_list: 0: id, 1: score
    :param new_item:
    :return:
    """
    index = search_position(rank_list, new_item)
    # search the position to insert into
    rank_list.insert(index, new_item)
    return index


# O(logN) search the position to insert into
def search_position(rank_list_m: list, new_item: List):
    if len(rank_list_m) == 0:
        return 0
    left = 0
    right = len(rank_list_m) - 1
    while left + 1 < right:
        mid = int((left + right) / 2)
        if rank_list_m[mid][1] <= new_item[1]:
            left = mid
        else:
            right = mid

    # consider the time.
    if rank_list_m[right][1] <= new_item[1]:
        return right + 1
    elif rank_list_m[left][1] <= new_item[1]:
        return left + 1
    else:
        return left


def generate_global_rank(ml_data_score_dic: dict, alg_name_list: List) -> dict:
    """
    ml_data_score_dic: { model_id: {alg: score1, alg2: score2} }
    return: { model_id: {alg1_alg2: rank_score} }
    """

    history = {}
    for alg in alg_name_list:
            history[alg] = []

    for arch_id, arch_score in ml_data_score_dic.items():
        # add model and score to local list
        for alg, score in arch_score.items():
            if alg in alg_name_list:
                binary_insert_get_rank(history[alg], [str(arch_id), float(score)])

    # convert multiple scores into rank value
    model_new_rank_score = {}
    current_explored_models = 0
    for alg in alg_name_list:
        current_explored_models = len(history[alg])
        for rank_index in range(len(history[alg])):
            ms_ins = history[alg][rank_index]
            # rank = index + 1, since index can be 0
            if ms_ins[0] in model_new_rank_score:
                model_new_rank_score[ms_ins[0]] += rank_index + 1
            else:
                model_new_rank_score[ms_ins[0]] = rank_index + 1

    for ele in model_new_rank_score.keys():
        model_new_rank_score[ele] = \
            {"_".join(list(alg_name_list)): model_new_rank_score[ele] / current_explored_models}

    return model_new_rank_score


def log_scale_x_array(num_points, max_minute, base=10, min_val=1) -> list:
    """
    return a list of mins in log scale distance.
    """
    # Set the minimum and maximum values for the log scale
    min_val = min_val  # 1 second
    max_val = max_minute * 60  # 1440 minutes converted to seconds

    # Generate the log scale values
    log_vals = np.logspace(np.log10(min_val), np.log10(max_val), num=num_points, base=base)

    # Convert the log scale values to minutes
    log_vals_min = log_vals / 60

    # Print the log scale values in minutes

    return log_vals_min.tolist()


def sample_in_log_scale(lst: List, num_points: int) -> List:
    indices = np.logspace(0, np.log10(len(lst) - 1), num_points + num_points // 2, dtype=int)
    # Remove any duplicate indices
    indices = np.unique(indices)
    return list(indices)


def sample_in_log_scale_new(lstM: List, num_points: int) -> List:
    lst = np.array(lstM)
    # Create an evenly spaced array in the log scale domain
    evenly_spaced_log_x = np.linspace(np.log10(lst.min()), np.log10(lst.max()), num_points)
    # Convert the new array back to the original scale
    evenly_spaced_x = 10 ** evenly_spaced_log_x
    # Find the indices of the sampled points in the original x-array
    indices = [np.abs(lst - point).argmin() for point in evenly_spaced_x]
    return indices


def sample_in_line_scale(lst: List, num_points: int) -> List:
    indices = np.linspace(0, len(lst) - 1, num_points, dtype=int)
    # Remove any duplicate indices
    indices = np.unique(indices)
    return list(indices)

