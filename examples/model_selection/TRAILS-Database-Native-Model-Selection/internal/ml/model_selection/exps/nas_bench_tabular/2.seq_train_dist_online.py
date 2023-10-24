import argparse
import calendar
import json
import logging
import os
import time

from exps.shared_args import parse_arguments


def partition_list_by_worker_id(lst, num_workers=15):
    partitions = []
    for i in range(num_workers):
        partitions.append([])
    for idx, item in enumerate(lst):
        worker_id = idx % num_workers
        partitions[worker_id].append(item)
    return partitions


def start_one_worker(queue, args, worker_id, my_partition, search_space_ins, res):
    from src.tools.io_tools import write_json, read_json
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)

    os.environ.setdefault("log_file_name", f"{args.log_name}_{args.dataset}_wkid_{worker_id}_{ts}.log")
    # import logging
    logger = logging.getLogger(f"{args.dataset}_wkid_{worker_id}_{ts}")
    if not os.path.exists(f"./{args.log_folder}"):
        os.makedirs(f"./{args.log_folder}")
    handler = logging.FileHandler(f"./{args.log_folder}/{args.log_name}_{args.dataset}_wkid_{worker_id}_{ts}.log")
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    from src.eva_engine.phase2.algo.trainer import ModelTrainer

    if args.total_models_per_worker is None:
        logger.info(
            f" ---- begin exploring, current worker have  "
            f"{len(my_partition)} models. explore all those models ")
    else:
        logger.info(f" ---- begin exploring, current worker have  "
                    f"{len(my_partition)} models. but explore {args.total_models_per_worker} models ")

    train_loader, val_loader, test_loader = queue.get()

    checkpoint_file_name = f"./base_line_res_{args.dataset}/train_baseline_{args.dataset}_wkid_{worker_id}.json"
    visited = read_json(checkpoint_file_name)
    if visited == {}:
        visited = {args.dataset: {}}
        logger.info(f" ---- initialize checkpointing with {visited} . ")
    else:
        logger.info(f" ---- recovery from checkpointing with {len(visited[args.dataset])} model. ")

    explored_arch_num = 0
    for arch_index in my_partition:
        print(f"begin to train the {arch_index}")
        model = search_space_ins.new_architecture(res[arch_index]).to(args.device)
        valid_auc, total_run_time, train_log = ModelTrainer.fully_train_arch(
            model=model,
            use_test_acc=False,
            epoch_num=args.epoch,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            args=args, logger=logger)

        logger.info(f' ----- model id: {res[arch_index]}, Val_AUC : {valid_auc} Total running time: '
                    f'{total_run_time}-----')

        # update the shared model eval res
        logger.info(f" ---- exploring {explored_arch_num} model. ")
        logger.info(f" ---- info: {json.dumps({res[arch_index]: train_log})}")
        visited[args.dataset][res[arch_index]] = train_log
        explored_arch_num += 1

        if args.total_models_per_worker is not None and explored_arch_num > args.total_models_per_worker:
            break

        logger.info(f" Saving result to: {checkpoint_file_name}")
        write_json(checkpoint_file_name, visited)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    args = parse_arguments()

    # set the log name
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)

    os.environ.setdefault("log_file_name", f"{args.log_name}_{args.dataset}_main_{ts}.log")
    os.environ.setdefault("base_dir", args.base_dir)

    from src.search_space.init_search_space import init_search_space
    from src.dataset_utils.structure_data_loader import libsvm_dataloader
    from src.tools.io_tools import write_json, read_json
    import torch.multiprocessing as mp

    search_space_ins = init_search_space(args)
    search_space_ins.load()

    # 1. main process partition data and group results,
    res = read_json(args.pre_partitioned_file)

    total_workers = args.worker_each_gpu * args.gpu_num
    all_partition = partition_list_by_worker_id(list(res.keys()), total_workers)

    train_loader, val_loader, test_loader = libsvm_dataloader(
        args=args,
        data_dir=os.path.join(args.base_dir, "data", "structure_data", args.dataset),
        nfield=args.nfield,
        batch_size=args.batch_size)

    # 2. put the shared dataloader into the queue,
    queue = mp.Queue()

    # 3. Create a list of processes to train the models
    processes = []
    worker_id = 0
    for gpu_id in range(args.gpu_num):
        for _ in range(args.worker_each_gpu):
            if args.device != "cpu":
                args.device = f"cuda:{gpu_id}"
            print(f"running process {[args.device, worker_id, len(all_partition[worker_id])]}")
            p = mp.Process(
                target=start_one_worker,
                args=(queue, args, worker_id, all_partition[worker_id], search_space_ins, res,
                      )
            )
            p.start()
            processes.append(p)
            worker_id += 1

    # 4. send to the queue
    for gpu_id in range(args.gpu_num):
        for _ in range(args.worker_each_gpu):
            print("putting to queue ....")
            queue.put((train_loader, val_loader, test_loader))

    print("All processing are running, waiting all to finish....")
    for p in processes:
        p.join()


