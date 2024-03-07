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

import os
from src.common.constant import Config
from src.tools.compute import generate_global_rank
from src.tools.io_tools import read_json

base_dir = os.environ.get("base_dir")
if base_dir is None: base_dir = os.getcwd()
print("base_dir is {}".format(base_dir))

# todo: move all those to a config file
# MLP related ground truth
mlp_train_frappe = os.path.join(base_dir, "tab_data/frappe/all_train_baseline_frappe.json")
mlp_train_uci_diabetes = os.path.join(base_dir, "tab_data/uci_diabetes/all_train_baseline_uci_160k_40epoch.json")
mlp_train_criteo = os.path.join(base_dir, "tab_data/criteo/all_train_baseline_criteo.json")

# score result
mlp_score_frappe = os.path.join(base_dir, "tab_data/frappe/score_frappe_batch_size_32_local_finish_all_models.json")
# mlp_score_frappe = os.path.join(base_dir, "tab_data/frappe/score_frappe_batch_size_32_nawot_synflow.json")
mlp_score_uci = os.path.join(base_dir, "tab_data/uci_diabetes/score_uci_diabetes_batch_size_32_all_metrics.json")
mlp_score_criteo = os.path.join(base_dir, "tab_data/criteo/score_criteo_batch_size_32.json")

#  0.8028456677612497
# todo: here is for debug expressFlow only
exp_mlp_score_frappe = os.path.join(base_dir, "micro_sensitivity/3_batch_size/4/score_mlp_sp_frappe_batch_size_32_cpu.json")
exp_mlp_score_uci = os.path.join(base_dir, "micro_sensitivity/3_batch_size/4/score_mlp_sp_uci_diabetes_batch_size_32_cpu.json")
exp_mlp_score_criteo = os.path.join(base_dir, "micro_sensitivity/3_batch_size/4/score_mlp_sp_criteo_batch_size_32_cpu.json")

# todo here we use weigth sharing.
mlp_score_frappe_weight_share = os.path.join(base_dir, "tab_data/weight_share_nas_frappe.json")

# pre computed result
score_one_model_time_dict = {
    "cpu": {
        Config.Frappe: 0.0211558125,
        Config.UCIDataset: 0.015039052631578948,
        Config.Criteo: 0.6824370454545454
    },
    "gpu": {
        Config.Frappe: 0.013744457142857143,
        Config.UCIDataset: 0.008209692307692308,
        Config.Criteo: 0.6095493157894737
    }
}

train_one_epoch_time_dict = {
    "cpu": {
        Config.Frappe: 5.122203075885773,
        Config.UCIDataset: 4.16297769,
        Config.Criteo: 422
    },
    "gpu": {
        Config.Frappe: 2.8,
        Config.UCIDataset: 1.4,
        Config.Criteo: 125
    }
}


class GTMLP:
    _instances = {}
    # use those algoroithm => new tfmem
    default_alg_name_list = ["nas_wot", "synflow"]
    device = "cpu"

    def __new__(cls, dataset: str):
        if dataset not in cls._instances:
            instance = super(GTMLP, cls).__new__(cls)
            instance.dataset = dataset
            if dataset == Config.Frappe:
                instance.mlp_train_path = mlp_train_frappe
                instance.mlp_score_path = mlp_score_frappe
                instance.mlp_score_path_expressflow = exp_mlp_score_frappe
                instance.mlp_score_path_weight_share = mlp_score_frappe_weight_share
            elif dataset == Config.Criteo:
                instance.mlp_train_path = mlp_train_criteo
                instance.mlp_score_path = mlp_score_criteo
                instance.mlp_score_path_expressflow = exp_mlp_score_criteo
                instance.mlp_score_path_weight_share = "./not_exist"
            elif dataset == Config.UCIDataset:
                instance.mlp_train_path = mlp_train_uci_diabetes
                instance.mlp_score_path = mlp_score_uci
                instance.mlp_score_path_expressflow = exp_mlp_score_uci
                instance.mlp_score_path_weight_share = "./not_exist"
            instance.mlp_train = read_json(instance.mlp_train_path)
            instance.mlp_score = read_json(instance.mlp_score_path)

            # todo: here we combine two json dict, remove later
            mlp_score_expressflow = read_json(instance.mlp_score_path_expressflow)
            for arch_id in mlp_score_expressflow:
                if arch_id in instance.mlp_score:
                    instance.mlp_score[arch_id].update(mlp_score_expressflow[arch_id])

            mlp_score_weight_share = read_json(instance.mlp_score_path_weight_share)
            for arch_id in mlp_score_weight_share:
                if arch_id in instance.mlp_score:
                    instance.mlp_score[arch_id].update({"weight_share": mlp_score_weight_share[arch_id]})

            instance.mlp_global_rank = generate_global_rank(
                instance.mlp_score, instance.default_alg_name_list)

            cls._instances[dataset] = instance
        return cls._instances[dataset]

    def get_all_trained_model_ids(self):
        return list(self.mlp_train[self.dataset].keys())

    def get_all_scored_model_ids(self):
        return list(self.mlp_score.keys())

    def get_score_one_model_time(self, device: str):
        _train_time_per_epoch = score_one_model_time_dict[device].get(self.dataset)
        if _train_time_per_epoch is None:
            raise NotImplementedError
        return _train_time_per_epoch

    def get_train_one_epoch_time(self, device: str):
        _train_time_per_epoch = train_one_epoch_time_dict[device].get(self.dataset)
        if _train_time_per_epoch is None:
            raise NotImplementedError
        return _train_time_per_epoch

    def get_valid_auc(self, arch_id: str, epoch_num: int):
        # todo: due to the too many job contention on server, the time usage may not valid.
        # train on gpu,
        time_usage = (int(epoch_num) + 1) * self.get_train_one_epoch_time("gpu")
        if self.dataset == Config.Frappe:
            if epoch_num is None or epoch_num >= 13: epoch_num = 13
            t_acc = self.mlp_train[self.dataset][arch_id][str(epoch_num)]["valid_auc"]
            time_usage = self.mlp_train[self.dataset][arch_id][str(epoch_num)]["train_val_total_time"]
            return t_acc, time_usage
        elif self.dataset == Config.Criteo:
            if epoch_num is None or epoch_num >= 10: epoch_num = 9
            t_acc = self.mlp_train[self.dataset][arch_id][str(epoch_num)]["valid_auc"]
            time_usage = self.mlp_train[self.dataset][arch_id][str(epoch_num)]["train_val_total_time"]
            return t_acc, time_usage
        elif self.dataset == Config.UCIDataset:
            if epoch_num is None or epoch_num >= 40: epoch_num = 39
            t_acc = self.mlp_train[self.dataset][arch_id][str(0)]["valid_auc"]
            time_usage = self.mlp_train[self.dataset][arch_id][str(epoch_num)]["train_val_total_time"]
            return t_acc, time_usage
        else:
            raise NotImplementedError

    def api_get_score(self, arch_id: str) -> dict:
        score_dic = self.mlp_score[arch_id]
        return score_dic

    def get_global_rank_score(self, arch_id):
        return self.mlp_global_rank[arch_id]
