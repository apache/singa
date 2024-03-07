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
import numpy as np
from src.common.constant import Config
from src.tools.io_tools import read_json

base_dir_folder = os.environ.get("base_dir")
if base_dir_folder is None:base_dir_folder = os.getcwd()
base_dir = os.path.join(base_dir_folder, "img_data")

print("gt_api running at {}".format(base_dir))
train_base201_c10 = os.path.join(base_dir, "train_based_201_c10.json")
train_base201_c100 = os.path.join(base_dir, "train_based_201_c100.json")
train_base201_img = os.path.join(base_dir, "train_based_201_img.json")

train_base101_c10 = os.path.join(base_dir, "train_based_101_c10_100run_24k_models.json")


def post_processing_train_base_result(search_space, dataset, x_max_value: int = None):

    if search_space == Config.NB201 and dataset == Config.c10:
        data = read_json(train_base201_c10)

    elif search_space == Config.NB201 and dataset == Config.c100:
        data = read_json(train_base201_c100)
    elif search_space == Config.NB201 and dataset == Config.imgNet:
        data = read_json(train_base201_img)

    elif search_space == Config.NB101 and dataset == Config.c10:
        data = read_json(train_base101_c10)
    else:
        print(f"Cannot read dataset {dataset} of file")
        raise

    # data is in form of
    """
    data[run_id] = {}
    data[run_id]["arch_id_list"]
    data[run_id]["current_best_acc"] 
    data[run_id]["x_axis_time"] 
    """

    acc_got_row = []
    time_used_row = []
    min_arch_across_all_run = 15625
    for run_id in data:
        acc_got_row.append(data[run_id]["current_best_acc"])
        time_used_row.append(data[run_id]["x_axis_time"])
        if len(data[run_id]["current_best_acc"]) < min_arch_across_all_run:
            min_arch_across_all_run = len(data[run_id]["current_best_acc"])

    # for each run, only use min_arch_across_all_run
    for i in range(len(acc_got_row)):
        acc_got_row[i] = acc_got_row[i][:min_arch_across_all_run]
        time_used_row[i] = time_used_row[i][:min_arch_across_all_run]

    acc_got = np.array(acc_got_row)
    time_used = np.array(time_used_row)

    if data['0']["current_best_acc"][-1] < 1:
        acc_got = acc_got * 100

    acc_l = np.quantile(acc_got, 0.25, axis=0)
    acc_m = np.quantile(acc_got, 0.5, axis=0)
    acc_h = np.quantile(acc_got, 0.75, axis=0)

    time_l = np.quantile(time_used, 0.25, axis=0)
    time_m = np.quantile(time_used, 0.5, axis=0).tolist()
    time_h = np.quantile(time_used, 0.75, axis=0)

    x_list = [ele/60 for ele in time_m]
    y_list_low = acc_l[:len(x_list)]
    y_list_m = acc_m[:len(x_list)]
    y_list_high = acc_h[:len(x_list)]

    # if the x array max value is provided.
    if x_max_value is not None:
        final_x_list = []
        final_x_list_low = []
        final_x_list_m = []
        final_x_list_high = []
        for i in range(len(x_list)):
            if x_list[i] <= x_max_value:
                final_x_list.append(x_list[i])
                final_x_list_low.append(y_list_low[i])
                final_x_list_m.append(y_list_m[i])
                final_x_list_high.append(y_list_high[i])
            else:
                break
        return final_x_list, final_x_list_low, final_x_list_m, final_x_list_high
    else:
        return x_list, y_list_low.tolist(), y_list_m.tolist(), y_list_high.tolist()


if __name__ == "__main__":
    search_space = Config.NB201
    dataset = Config.c100
    x_list, y_list_low, y_list_m, y_list_high = post_processing_train_base_result(search_space, dataset)

    from matplotlib import pyplot as plt

    plt.fill_between(x_list, y_list_low, y_list_high, alpha=0.1)
    plt.plot(x_list, y_list_m, "-*", label="Training-based")

    plt.xscale("symlog")
    plt.grid()
    plt.xlabel("Time Budget given by user (mins)")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.show()


