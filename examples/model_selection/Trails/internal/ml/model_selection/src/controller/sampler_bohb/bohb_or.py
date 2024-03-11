
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

import os, sys, time, random, argparse, collections
from src.tools.env_tools import prepare_seed
from src.logger import logger
from models import CellStructure, get_search_spaces

# BOHB: Robust and Efficient Hyperparameter Optimization at Scale, ICML 2018
import ConfigSpace
from hpbandster.optimizers.bohb import BOHB
import hpbandster.core.nameserver as hpns
from hpbandster.core.worker import Worker

from nats_bench import create



def time_string():
    ISOTIMEFORMAT = "%Y-%m-%d %X"
    string = "[{:}]".format(time.strftime(ISOTIMEFORMAT, time.gmtime(time.time())))
    return string


def get_topology_config_space(search_space, max_nodes=4):
    cs = ConfigSpace.ConfigurationSpace()
    # edge2index   = {}
    for i in range(1, max_nodes):
        for j in range(i):
            node_str = "{:}<-{:}".format(i, j)
            cs.add_hyperparameter(
                ConfigSpace.CategoricalHyperparameter(node_str, search_space)
            )
    return cs


def get_size_config_space(search_space):
    cs = ConfigSpace.ConfigurationSpace()
    for ilayer in range(search_space["numbers"]):
        node_str = "layer-{:}".format(ilayer)
        cs.add_hyperparameter(
            ConfigSpace.CategoricalHyperparameter(node_str, search_space["candidates"])
        )
    return cs


def config2topology_func(max_nodes=4):
    def config2structure(config):
        genotypes = []
        for i in range(1, max_nodes):
            xlist = []
            for j in range(i):
                node_str = "{:}<-{:}".format(i, j)
                op_name = config[node_str]
                xlist.append((op_name, j))
            genotypes.append(tuple(xlist))
        return CellStructure(genotypes)

    return config2structure


def config2size_func(search_space):
    def config2structure(config):
        channels = []
        for ilayer in range(search_space["numbers"]):
            node_str = "layer-{:}".format(ilayer)
            channels.append(str(config[node_str]))
        return ":".join(channels)

    return config2structure


class MyWorker(Worker):
    def __init__(self, *args, convert_func=None, dataset=None, api=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.convert_func = convert_func
        self._dataset = dataset
        self._api = api
        self.total_times = []
        self.trajectory = []

    def compute(self, config, budget, **kwargs):
        arch = self.convert_func(config)
        accuracy, latency, time_cost, total_time = self._api.simulate_train_eval(
            arch, self._dataset, iepoch=int(budget) - 1, hp="12"
        )
        self.trajectory.append((accuracy, arch))
        self.total_times.append(total_time)
        return {"loss": 100 - accuracy, "info": self._api.query_index_by_arch(arch)}


def main(xargs, api):
    prepare_seed(xargs.rand_seed)

    logger.info("{:} use api : {:}".format(time_string(), api))
    api.reset_time()
    search_space = get_search_spaces(xargs.search_space, "nats-bench")
    if xargs.search_space == "tss":
        cs = get_topology_config_space(search_space)
        config2structure = config2topology_func()
    else:
        cs = get_size_config_space(search_space)
        config2structure = config2size_func(search_space)

    hb_run_id = "0"

    NS = hpns.NameServer(run_id=hb_run_id, host="localhost", port=0)
    ns_host, ns_port = NS.start()
    num_workers = 1

    workers = []
    for i in range(num_workers):
        w = MyWorker(
            nameserver=ns_host,
            nameserver_port=ns_port,
            convert_func=config2structure,
            dataset=xargs.dataset,
            api=api,
            run_id=hb_run_id,
            id=i,
        )
        w.run(background=True)
        workers.append(w)

    start_time = time.time()
    bohb = BOHB(
        configspace=cs,
        run_id=hb_run_id,
        eta=3,
        min_budget=1,
        max_budget=12,
        nameserver=ns_host,
        nameserver_port=ns_port,
        num_samples=xargs.num_samples,
        random_fraction=xargs.random_fraction,
        bandwidth_factor=xargs.bandwidth_factor,
        ping_interval=10,
        min_bandwidth=xargs.min_bandwidth,
    )

    results = run(xargs.n_iters, min_n_workers=num_workers)

    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    # print('There are {:} runs.'.format(len(results.get_all_runs())))
    # workers[0].total_times
    # workers[0].trajectory
    current_best_index = []
    for idx in range(len(workers[0].trajectory)):
        trajectory = workers[0].trajectory[: idx + 1]
        arch = max(trajectory, key=lambda x: x[0])[1]
        current_best_index.append(api.query_index_by_arch(arch))

    best_arch = max(workers[0].trajectory, key=lambda x: x[0])[1]
    logger.log(
        "Best found configuration: {:} within {:.3f} s".format(
            best_arch, workers[0].total_times[-1]
        )
    )
    info = api.query_info_str_by_arch(
        best_arch, "200" if xargs.search_space == "tss" else "90"
    )
    logger.log("{:}".format(info))
    logger.log("-" * 100)
    logger.close()

    return logger.log_dir, current_best_index, workers[0].total_times


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "BOHB: Robust and Efficient Hyperparameter Optimization at Scale"
    )
    parser.add_argument(
        "--dataset",
        default="cifar10",
        type=str,
        choices=["cifar10", "cifar100", "ImageNet16-120"],
        help="Choose between Cifar10/100 and ImageNet-16.",
    )
    # general arg
    parser.add_argument(
        "--search_space",
        default="tss",
        type=str,
        choices=["tss", "sss"],
        help="Choose the search space.",
    )
    parser.add_argument(
        "--time_budget",
        type=int,
        default=20000,
        help="The total time cost budge for searching (in seconds).",
    )
    parser.add_argument(
        "--loops_if_rand", type=int, default=500, help="The total runs for evaluation."
    )
    # BOHB
    parser.add_argument(
        "--strategy",
        default="sampling",
        type=str,
        nargs="?",
        help="optimization strategy for the acquisition function",
    )
    parser.add_argument(
        "--min_bandwidth",
        default=0.3,
        type=float,
        nargs="?",
        help="minimum bandwidth for KDE",
    )
    parser.add_argument(
        "--num_samples",
        default=64,
        type=int,
        nargs="?",
        help="number of samples for the acquisition function",
    )
    parser.add_argument(
        "--random_fraction",
        default=0.33,
        type=float,
        nargs="?",
        help="fraction of random configurations",
    )
    parser.add_argument(
        "--bandwidth_factor",
        default=3,
        type=int,
        nargs="?",
        help="factor multiplied to the bandwidth",
    )
    parser.add_argument(
        "--n_iters",
        default=300,
        type=int,
        nargs="?",
        help="number of iterations for optimization method",
    )
    # log
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./output/search",
        help="Folder to save checkpoints and log.",
    )
    parser.add_argument("--rand_seed", type=int, default=-1, help="manual seed")
    args = parser.parse_args()

    api = create(None, args.search_space, fast_mode=False, verbose=False)

    args.save_dir = os.path.join(
        "{:}-{:}".format(args.save_dir, args.search_space),
        "{:}-T{:}".format(args.dataset, args.time_budget),
        "BOHB",
    )
    print("save-dir : {:}".format(args.save_dir))

    if args.rand_seed < 0:
        save_dir, all_info = None, collections.OrderedDict()
        for i in range(args.loops_if_rand):
            print("{:} : {:03d}/{:03d}".format(time_string(), i, args.loops_if_rand))
            args.rand_seed = random.randint(1, 100000)
            save_dir, all_archs, all_total_times = main(args, api)
            all_info[i] = {"all_archs": all_archs, "all_total_times": all_total_times}
        save_path = save_dir / "results.pth"
        print("save into {:}".format(save_path))

        import pickle
        with open(save_path, 'wb') as f:
            pickle.dump(all_info, f)

    else:
        main(args, api)


