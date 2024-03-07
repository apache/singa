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

import json
import os
import sqlite3
import traceback

from src.common.constant import Config

base_folder_dir = os.environ.get("base_dir")
if base_folder_dir is None: base_folder_dir = os.getcwd()
base_dir = os.path.join(base_folder_dir, "img_data")
print("local api running at {}".format(base_dir))

# sum score is better
tf_smt_file_NB101C10 = os.path.join(base_dir, "TFMEM_101_c10_100run_8k_models_score_sum")
tf_smt_file_NB201C10 = os.path.join(base_dir, "TFMEM_201_c10_100run_score_sum")
tf_smt_file_NB201C100 = os.path.join(base_dir, "TFMEM_201_c100_100run_score_sum")
tf_smt_file_NB201Img = os.path.join(base_dir, "TFMEM_201_imgNet_100run_score_sum")

# rank is not as good as sum
# tf_smt_file_NB201C10 = os.path.join(base_dir, "TFMEM_201_c10_100run_rank_bugs")
# tf_smt_file_NB201C100 = os.path.join(base_dir, "TFMEM_201_c100_200run_rank")
# tf_smt_file_NB201Img = os.path.join(base_dir, "TFMEM_201_imgNet_200run_rank")

con = None
cur = None


# fetch result from simulated result
def fetch_from_db(space_name, dataset, run_id_m, N_m):
    """
    :param run_id_m: run_id 100 max
    :param B1_m: number of models evaluted
    :return:
    """
    global con
    global cur
    if con is None:
        if space_name == Config.NB201:
            if dataset == Config.c10:
                tf_smt_used = tf_smt_file_NB201C10
            elif dataset == Config.c100:
                tf_smt_used = tf_smt_file_NB201C100
            elif dataset == Config.imgNet:
                tf_smt_used = tf_smt_file_NB201Img
            else:
                print(f"{dataset} is Not implemented")
                raise
        elif space_name == Config.NB101:
            if dataset == Config.c10:
                tf_smt_used = tf_smt_file_NB101C10
            else:
                print(f"{dataset}Not implemented")
                raise
        else:
            print(f"{space_name} is Not implemented")
            raise

        print(tf_smt_used)
        con = sqlite3.connect(tf_smt_used)
        cur = con.cursor()

    res = cur.execute(
        "SELECT * FROM simulateExp WHERE run_num = {} and model_explored = {}".format(run_id_m, N_m))
    fetch_res = res.fetchone()

    try:
        arch_id = fetch_res[2]
        candidates = json.loads(fetch_res[3])
        current_time = float(fetch_res[4])
    except:
        print(traceback.format_exc())
        raise f"res is None when using run_id ={run_id_m} and bm = {N_m}"

    return arch_id, candidates, current_time


if __name__ == '__main__':
    print(fetch_from_db(Config.NB201, Config.c100, 3, 10))
