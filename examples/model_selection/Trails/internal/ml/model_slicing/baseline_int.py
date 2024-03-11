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
import torch
import argparse
from model_slicing.algorithm.src.model.sparsemax_verticalMoe import SliceModel, SparseMax_VerticalSAMS
import time
import psycopg2
from model_slicing.algorithm.src.model.factory import initialize_model
from typing import Any, List, Dict, Tuple
import json

USER = "postgres"
HOST = "localhost"
PORT = "28814"
DB_NAME = "pg_extension"
PASSWOD = "1234"

time_dict = {
    "load_model": 0,
    "data_query_time": 0,
    "py_conver_to_tensor": 0,
    "tensor_to_gpu": 0,
    "py_compute": 0

}


def read_json(file_name):
    print(f"Loading {file_name}...")
    is_exist = os.path.exists(file_name)
    if is_exist:
        with open(file_name, 'r') as readfile:
            data = json.load(readfile)
        return data
    else:
        print(f"{file_name} is not exist")
        return {}


def fetch_and_preprocess(conn, batch_size, database):
    cur = conn.cursor()
    # Select rows greater than last_id
    cur.execute(f"SELECT * FROM {database}_int_train LIMIT {batch_size}")
    rows = cur.fetchall()
    return rows


def pre_processing(mini_batch_data: List[Tuple]):
    """
    mini_batch_data: [('0', '0', '123:123', '123:123', '123:123',)
    """
    feat_id = torch.LongTensor(mini_batch_data)
    print("feat_id = ", feat_id[:, 2:].size())
    return {'id': feat_id[:, 2:]}


def fetch_data(database, batch_size):
    global time_dict
    print("Data fetching ....")
    begin_time = time.time()
    with psycopg2.connect(database=DB_NAME, user=USER, host=HOST, port=PORT) as conn:
        rows = fetch_and_preprocess(conn, batch_size, database)
    time_dict["data_query_time"] += time.time() - begin_time
    print(f"Data fetching done {rows[0]}, size = {len(rows)}, type = {type(rows)}, {type(rows[0])}")

    print("Data preprocessing ....")
    begin_time = time.time()
    batch = pre_processing(rows)
    time_dict["py_conver_to_tensor"] += time.time() - begin_time
    print("Data preprocessing done")
    return batch


def load_model(tensorboard_path: str, device: str = "cuda"):
    """
    Args:
    tensorboard_path: the path of the directory of tensorboard
    """
    arg_file_path = os.path.join(tensorboard_path, "args.txt")
    model_config = reload_argparse(arg_file_path)

    net = initialize_model(model_config)

    model_pth_path = os.path.join(tensorboard_path, "best_model.pth")
    saved_state_dict = torch.load(model_pth_path, map_location=device)

    net.load_state_dict(saved_state_dict)
    print("successfully load model")
    return net, model_config


def if_cuda_avaiable(device):
    if "cuda" in device:
        return True
    else:
        return False


def reload_argparse(file_path: str):
    d = {}

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            key, value = line.strip('\n').split(',')
            # print(f"{key}, {value}\n")
            try:
                re = eval(value)
            except:
                re = value
            d[key] = re

    return argparse.Namespace(**d)


parser = argparse.ArgumentParser(description='predict FLOPS')
parser.add_argument('path', type=str,
                    help="directory to model file")
parser.add_argument('--flag', '-p', action='store_true',
                    help="wehther to print profile")
parser.add_argument('--print_net', '--b', action='store_true',
                    help="print the structure of network")

parser.add_argument('--device', type=str, default="cuda")
parser.add_argument('--dataset', type=str, default="frappe")
parser.add_argument('--target_batch', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=10000)
parser.add_argument('--col_cardinalities_file', type=str, default="path to the stored file")

if __name__ == '__main__':
    args = parser.parse_args()
    path = args.path
    flag = args.flag
    device = torch.device(args.device)
    print(path)
    load_time = time.time()
    net, config = load_model(path, args.device)
    net: SparseMax_VerticalSAMS = net
    config.workload = 'random'
    time_dict["load_model"] = time.time() - load_time

    print(config.workload)

    overall_query_latency = time.time()
    if config.net == "sparsemax_vertical_sams":
        alpha = net.sparsemax.alpha
        print(alpha)

    print()

    col_cardinalities = read_json(args.col_cardinalities_file)
    target_sql = torch.tensor([col[-1] for col in col_cardinalities]).reshape(1, -1)

    net.eval()
    net = net.to(device)
    with torch.no_grad():
        sql = target_sql.to(device)
        if config.net == "sparsemax_vertical_sams":
            subnet: SliceModel = net.tailor_by_sql(sql)
            subnet.to(device)
        else:
            subnet = net
        subnet.eval()
        target_list, y_list = [], []
        ops = 0

        # default batch to 1024
        num_ite = args.target_batch // args.batch_size
        print(f"num_ite = {num_ite}")
        for i in range(num_ite):
            # fetch from db
            data_batch = fetch_data(args.dataset, args.batch_size)
            print("Copy to device")
            # wait for moving data to GPU
            begin = time.time()
            x_id = data_batch['id'].to(device)
            if if_cuda_avaiable(args.device):
                torch.cuda.synchronize()
            time_dict["tensor_to_gpu"] += time.time() - begin

            print(f"begin to compute on {args.device}, is_cuda = {if_cuda_avaiable(args.device)}")
            # compute
            begin = time.time()
            y = subnet(x_id, None)
            if if_cuda_avaiable(args.device):
                torch.cuda.synchronize()
            time_dict["py_compute"] += time.time() - begin
    time_dict["overall_query_latency"] = time.time() - overall_query_latency
    print(time_dict)
