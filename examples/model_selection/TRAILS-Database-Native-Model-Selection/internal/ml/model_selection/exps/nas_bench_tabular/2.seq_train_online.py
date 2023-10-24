import calendar
import json
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


if __name__ == "__main__":

    args = parse_arguments()

    # set the log name
    gmt = time.gmtime()
    ts = calendar.timegm(gmt)

    os.environ.setdefault("log_logger_folder_name", f"{args.log_folder}")
    os.environ.setdefault("log_file_name", f"{args.log_name}_{args.dataset}_wkid_{args.worker_id}_{ts}.log")
    os.environ.setdefault("base_dir", args.base_dir)

    from src.logger import logger
    from src.eva_engine.phase2.algo.trainer import ModelTrainer
    from src.search_space.init_search_space import init_search_space
    from src.dataset_utils.structure_data_loader import libsvm_dataloader
    from src.tools.io_tools import write_json, read_json

    search_space_ins = init_search_space(args)
    search_space_ins.load()

    # 1. data loader
    logger.info(f" Loading data....")
    train_loader, val_loader, test_loader = libsvm_dataloader(
        args=args,
        data_dir=os.path.join(args.base_dir, "data", "structure_data", args.dataset),
        nfield=args.nfield,
        batch_size=args.batch_size)

    res = read_json(args.pre_partitioned_file)

    all_partition = partition_list_by_worker_id(list(res.keys()), args.total_workers)

    if args.total_models_per_worker == -1:
        logger.info(
            f" ---- begin exploring, current worker have  "
            f"{len(all_partition[args.worker_id])} models. explore all those models ")
    else:
        logger.info(f" ---- begin exploring, current worker have  "
                    f"{len(all_partition[args.worker_id])} models. but explore {args.total_models_per_worker} models ")

    # read the checkpoint
    checkpoint_file_name = f"{args.result_dir}/train_baseline_{args.dataset}_wkid_{args.worker_id}.json"
    visited = read_json(checkpoint_file_name)
    if visited == {}:
        visited = {args.dataset: {}}
        logger.info(f" ---- initialize checkpointing with {visited} . ")
    else:
        logger.info(f" ---- recovery from checkpointing with {len(visited[args.dataset])} model. ")

    explored_arch_num = 0
    for arch_index in all_partition[args.worker_id]:
        print(f"begin to train the {arch_index}")
        if res[arch_index] in visited[args.dataset]:
            logger.info(f" ---- model {res[arch_index]} already visited")
            continue
        model = search_space_ins.new_architecture(res[arch_index])
        model.init_embedding(requires_grad=True)
        model.to(args.device)
        valid_auc, total_run_time, train_log = ModelTrainer.fully_train_arch(
            model=model,
            use_test_acc=False,
            epoch_num=args.epoch,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            args=args)

        logger.info(f' ----- model id: {res[arch_index]}, Val_AUC : {valid_auc} Total running time: '
                    f'{total_run_time}-----')

        # update the shared model eval res
        logger.info(f" ---- exploring {explored_arch_num} model. ")
        logger.info(f" ---- info: {json.dumps({res[arch_index]: train_log})}")
        visited[args.dataset][res[arch_index]] = train_log
        explored_arch_num += 1

        if args.total_models_per_worker != -1 and explored_arch_num > args.total_models_per_worker:
            break

        logger.info(f" Saving result to: {checkpoint_file_name}")
        write_json(checkpoint_file_name, visited)
