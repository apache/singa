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
import psutil
import gpustat
import threading
import time
from src.tools.io_tools import write_json
import sys
import torch


def get_variable_memory_size(obj):
    # If it's a PyTorch tensor and on the GPU
    if torch.is_tensor(obj) and obj.is_cuda:
        return obj.element_size() * obj.nelement()
    else:
        return sys.getsizeof(obj)


def print_cpu_gpu_usage(interval=1, output_file="path_to_folder", stop_event=None):
    def print_usage():
        print("Starting to print usage")  # Debugging print
        # Get current process
        main_process = psutil.Process(os.getpid())

        # Create an empty dictionary to store metrics
        metrics = {'cpu_usage': [], 'memory_usage': [], 'gpu_usage': []}

        while not stop_event.is_set():
            cpu_percent = 0
            mem_usage_mb = 0
            main_process.cpu_percent()
            for process in main_process.children(recursive=True):  # Include all child processes
                try:
                    cpu_percent += process.cpu_percent()
                    mem_usage_mb += process.memory_info().rss / (1024 ** 2)
                except psutil.NoSuchProcess:
                    # Process does not exist, so add 0 to cpu_percent and mem_usage_mb
                    pass
            cpu_percent += main_process.cpu_percent()
            mem_usage_mb += main_process.memory_info().rss / (1024 ** 2)

            metrics['cpu_usage'].append(cpu_percent)
            metrics['memory_usage'].append(mem_usage_mb)

            try:
                gpu_stats = gpustat.GPUStatCollection.new_query()
                for gpu in gpu_stats:
                    metrics['gpu_usage'].append((gpu.index, gpu.utilization, gpu.memory_used))
            except Exception as e:
                pass
                # print(f"Exception encountered when fetching GPU stats: {e}")

            # If it's time to write metrics to a file, do so
            if len(metrics['cpu_usage']) % 40 == 0:
                write_json(output_file, metrics)

            time.sleep(interval)

        print("Stop monitering, flust to disk")
        write_json(output_file, metrics)

    stop_event = stop_event or threading.Event()
    thread = threading.Thread(target=print_usage)
    thread.start()
    return stop_event, thread


def print_memory_usage():
    # Get current process
    main_process = psutil.Process(os.getpid())
    # Create an empty dictionary to store metrics
    metrics = {'cpu_usage': [], 'memory_usage': []}
    cpu_percent = 0
    mem_usage_mb = 0
    main_process.cpu_percent()
    for process in main_process.children(recursive=True):  # Include all child processes
        try:
            cpu_percent += process.cpu_percent()
            mem_usage_mb += process.memory_info().rss / (1024 ** 2)
        except psutil.NoSuchProcess:
            # Process does not exist, so add 0 to cpu_percent and mem_usage_mb
            pass
    cpu_percent += main_process.cpu_percent()
    mem_usage_mb += main_process.memory_info().rss / (1024 ** 2)
    metrics['cpu_usage'].append(cpu_percent)
    metrics['memory_usage'].append(mem_usage_mb)
    print(metrics)
