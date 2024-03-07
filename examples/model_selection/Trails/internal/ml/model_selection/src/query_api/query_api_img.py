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
import random
from src.common.constant import Config
from src.tools.io_tools import read_json, write_json
from src.query_api.singleton import Singleton
from src.tools.io_tools import read_pickle
from src.tools.compute import generate_global_rank


base_dir_folder = os.environ.get("base_dir")
if base_dir_folder is None: base_dir_folder = os.getcwd()
base_dir = os.path.join(base_dir_folder, "img_data")
print("local api running at {}".format(base_dir))

# todo: move all those to a config file
# score result
pre_score_path_101C10 = os.path.join(base_dir, "score_101_15k_c10_128.json")
pre_score_path_201C10 = os.path.join(base_dir, "score_201_15k_c10_bs32_ic16.json")
pre_score_path_201C100 = os.path.join(base_dir, "score_201_15k_c100_bs32_ic16.json")
pre_score_path_201IMG = os.path.join(base_dir, "score_201_15k_imgNet_bs32_ic16.json")

# expreflow
expreflow_score_path_101C10 = os.path.join(base_dir, "score_nasbench101_cifar10_batch_size_32_cpu.json")
# expreflow_score_path_201C10 = os.path.join(base_dir, "score_nasbench201_cifar10_batch_size_32_cpu.json")
# expreflow_score_path_201C100 = os.path.join(base_dir, "score_nasbench201_cifar100_batch_size_32_cpu.json")
# expreflow_score_path_201IMG = os.path.join(base_dir, "score_nasbench201_ImageNet16-120_batch_size_32_cpu.json")

expreflow_score_path_201C10 = os.path.join(base_dir_folder, "score_scale_traj_width/score_nasbench201_cifar10_batch_size_32_cpu.json")
expreflow_score_path_201C100 = os.path.join(base_dir_folder, "score_scale_traj_width/score_nasbench201_cifar100_batch_size_32_cpu.json")
expreflow_score_path_201IMG = os.path.join(base_dir_folder, "score_scale_traj_width/score_nasbench201_ImageNet16-120_batch_size_32_cpu.json")

# training accuracy result.
gt201 = os.path.join(base_dir, "ground_truth/201_allEpoch_info")
gt101 = os.path.join(base_dir, "ground_truth/101_allEpoch_info_json")
gt101P = os.path.join(base_dir, "ground_truth/nasbench1_accuracy.p")
id_to_hash_path = os.path.join(base_dir, "ground_truth/nb101_id_to_hash.json")


# We pre-compute the time usage, and get a range,
# Then we randomly pick one value from the range each time
def guess_score_time(search_space_m, dataset):
    if search_space_m == Config.NB101:
        return Gt101.guess_score_time()
    if search_space_m == Config.NB201:
        return Gt201.guess_score_time(dataset)


def guess_train_one_epoch_time(search_space_m, dataset):
    if search_space_m == Config.NB101:
        return Gt101().guess_train_one_epoch_time()
    if search_space_m == Config.NB201:
        return Gt201().guess_train_one_epoch_time(dataset)
    raise NotImplementedError


class ImgScoreQueryApi:
    # Multiton pattern
    # use those algoroithm => new tfmem
    default_alg_name_list = ["nas_wot", "synflow"]
    _instances = {}

    def __new__(cls, search_space_name: str, dataset: str):
        if (search_space_name, dataset) not in cls._instances:
            instance = super(ImgScoreQueryApi, cls).__new__(cls)
            instance.search_space_name, instance.dataset = search_space_name, dataset

            # read pre-scored file path
            if search_space_name == Config.NB201:
                if dataset == Config.c10:
                    instance.pre_score_path = pre_score_path_201C10
                    instance.express_score_path = expreflow_score_path_201C10
                elif dataset == Config.c100:
                    instance.pre_score_path = pre_score_path_201C100
                    instance.express_score_path = expreflow_score_path_201C100
                elif dataset == Config.imgNet:
                    instance.pre_score_path = pre_score_path_201IMG
                    instance.express_score_path = expreflow_score_path_201IMG
            if search_space_name == Config.NB101:
                instance.pre_score_path = pre_score_path_101C10
                instance.express_score_path = expreflow_score_path_101C10

            instance.data = read_json(instance.pre_score_path)
            express_score_data = read_json(instance.express_score_path)
            for arch_id in express_score_data:
                if arch_id in instance.data:
                    instance.data[arch_id].update(express_score_data[arch_id])
                else:
                    instance.data[arch_id] = express_score_data[arch_id]

            instance.global_rank = generate_global_rank(
                instance.data, instance.default_alg_name_list)

            cls._instances[(search_space_name, dataset)] = instance
        return cls._instances[(search_space_name, dataset)]

    def api_get_score(self, arch_id: str, tfmem: str = None):
        # retrieve score from pre-scored file
        if tfmem is None:
            return self.data[arch_id]
        else:
            return {tfmem: float(self.data[arch_id][tfmem])}

    def update_existing_data(self, arch_id, alg_name, score_str):
        """
        Add new arch's inf into data
        :param arch_id:
        :param alg_name:
        :param score_str:
        :return:
        """
        if str(arch_id) not in self.data:
            self.data[str(arch_id)] = {}
        else:
            self.data[str(arch_id)] = self.data[str(arch_id)]
        self.data[str(arch_id)][alg_name] = '{:f}'.format(score_str)

    def is_arch_and_alg_inside_data(self, arch_id, alg_name):
        if arch_id in self.data and alg_name in self.data[arch_id]:
            return True
        else:
            return False

    def is_arch_inside_data(self, arch_id):
        if arch_id in self.data:
            return True
        else:
            return False

    def get_len_data(self):
        return len(self.data)

    def save_latest_data(self):
        """
        update the latest score data
        """
        write_json(self.pre_score_path, self.data)

    def get_all_scored_model_ids(self):
        return list(self.data.keys())

    def get_global_rank_score(self, arch_id):
        return self.global_rank[arch_id]


class Gt201(metaclass=Singleton):

    @classmethod
    def guess_score_time(cls, dataset=Config.c10):
        return random.randint(3315, 4502) * 0.0001

    def __init__(self):
        self.data201 = read_json(gt201)

    def get_c10valid_200epoch_test_info(self, arch_id: int):
        """
        cifar10-valid means train with train set, valid with validation dataset
        Thus, acc is lower than train with train+valid.
        :param arch_id:
        :return:
        """
        return self.query_200_epoch(str(arch_id), Config.c10_valid)

    def get_c10_200epoch_test_info(self, arch_id: int):
        """
        cifar10-valid means train with train set, valid with validation dataset
        Thus, acc is lower than train with train+valid.
        :param arch_id:
        :return:
        """
        return self.query_200_epoch(str(arch_id), Config.c10)

    def get_c100_200epoch_test_info(self, arch_id: int):
        return self.query_200_epoch(str(arch_id), Config.c100)

    def get_imgNet_200epoch_test_info(self, arch_id: int):
        return self.query_200_epoch(str(arch_id), Config.imgNet)

    def query_200_epoch(self, arch_id: str, dataset, epoch_num: int = 199):
        if epoch_num is None or epoch_num > 199:
            epoch_num = 199
        arch_id = str(arch_id)
        t_acc = self.data201[arch_id]["200"][dataset][str(epoch_num)]["test_accuracy"]
        time_usage = self.data201[arch_id]["200"][dataset][str(epoch_num)]["time_usage"]
        return t_acc, time_usage

    def query_12_epoch(self, arch_id: str, dataset, epoch_num: int = 11):
        if epoch_num is None or epoch_num > 11:
            epoch_num = 11
        arch_id = str(arch_id)
        t_acc = self.data201[arch_id]["12"][dataset][str(epoch_num)]["test_accuracy"]
        time_usage = self.data201[arch_id]["12"][dataset][str(epoch_num)]["time_usage"]
        return t_acc, time_usage

    def count_models(self):
        return len(self.data201)

    def guess_train_one_epoch_time(self, dataset):
        if dataset == Config.c10:
            dataset = Config.c10_valid
        # pick the max value over 5k arch training time, it's 40
        # res = 0
        # for arch_id in range(15624):
        #     _, time_usage = self.query_200_epoch(str(arch_id), dataset, 1)
        #     if time_usage > res:
        #         res = time_usage
        # return res
        arch_id = random.randint(1, 15625)
        time_usage = self.data201[str(arch_id)]["200"][dataset]["0"]["time_usage"]
        return time_usage

    def get_all_trained_model_ids(self):
        # 201 all data has the same model set.
        return list(self.data201.keys())


class Gt101(metaclass=Singleton):

    @classmethod
    def guess_score_time(cls):
        return random.randint(1169, 1372) * 0.0001

    def __init__(self):
        self.data101_from_zerocost = read_pickle(gt101P)
        self.id_to_hash_map = read_json(id_to_hash_path)
        self.data101_full = read_json(gt101)

    def get_c10_test_info(self, arch_id: str, dataset: str = Config.c10, epoch_num: int = 108):
        """
        Default use 108 epoch for c10, this is the largest epoch number.
        :param dataset:
        :param arch_id: architecture id
        :param epoch_num: query the result of the specific epoch number
        :return:
        """
        if dataset != Config.c10:
            raise "NB101 only have c10 results"

        if epoch_num is None or epoch_num > 108:
            epoch_num = 108
        elif epoch_num > 36:
            epoch_num = 36
        elif epoch_num > 12:
            epoch_num = 12
        elif epoch_num > 4:
            epoch_num = 4
        else:
            epoch_num = 4
        arch_id = str(arch_id)
        # this is acc from zero-cost paper, which only record 108 epoch' result [test, valid, train]
        # t_acc = self.data101_from_zerocost[self.id_to_hash_map[arch_id]][0]
        # this is acc from parse_testacc_101.py,
        t_acc = self.data101_full[arch_id][Config.c10][str(epoch_num)]["test-accuracy"]
        time_usage = self.data101_full[arch_id][Config.c10][str(epoch_num)]["time_usage"]
        # print(f"[Debug]: Acc different = {t_acc_usage - t_acc}")
        return t_acc, time_usage

    def count_models(self):
        return len(self.data101_from_zerocost)

    def guess_train_one_epoch_time(self):
        # only have information for 4 epoch
        d = dict.fromkeys(self.data101_full)
        keys = random.sample(list(d), 15000)

        # pick the max value over 5k arch training time
        res = 0
        for rep_time in range(15000):
            arch_id = keys[rep_time]
            _, time_usage = self.get_c10_test_info(arch_id=arch_id, dataset=Config.c10, epoch_num=4)
            if time_usage > res:
                res = time_usage
        return res

    def get_all_trained_model_ids(self):
        return list(self.data101_full.keys())


if __name__ == "__main__":
    lapi = ImgScoreQueryApi(Config.NB101, Config.c10)
    lapi.get_len_data()
