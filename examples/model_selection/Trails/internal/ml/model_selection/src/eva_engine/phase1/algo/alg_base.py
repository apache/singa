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


import math
import time
from abc import abstractmethod
import torch
from torch import nn


class Evaluator:
    def __init__(self):
        pass

    @abstractmethod
    def evaluate(self, arch: nn.Module,
                 device: str,
                 batch_data: object, batch_labels: torch.Tensor,
                 space_name: str
                 ) -> float:
        """
        Score each architecture with predefined architecture and data
        :param arch: architecture to be scored
        :param device:  cpu or gpu
        :param batch_data: a mini batch of data, [ batch_size, channel, W, H ] or dict for structure data
        :param batch_labels: a mini batch of labels
        :param space_name: string
        :return: score
        """
        raise NotImplementedError

    def evaluate_wrapper(self, arch, device: str, space_name: str,
                         batch_data: torch.tensor,
                         batch_labels: torch.tensor) -> (float, float):
        """
        :param arch: architecture to be scored
        :param device: cpu or GPU
        :param space_name: search space name
        :param batch_data: a mini batch of data, [ batch_size, channel, W, H ]
        :param batch_labels: a mini batch of labels
        :return: score, timeUsage
        """

        arch.train()
        # arch.zero_grad()

        # measure scoring time
        if "cuda" in device:
            torch.cuda.synchronize()
            # use this will not need cuda.sync
            # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            # starter.record()
            starter, ender = time.time(), time.time()
        else:
            starter, ender = time.time(), time.time()

        # score
        score = self.evaluate(arch, device, batch_data, batch_labels, space_name)

        if "cuda" in device:
            # ender.record()
            # implicitly waits for the event to be marked as complete before calculating the time difference
            # curr_time = starter.elapsed_time(ender)
            torch.cuda.synchronize()
            ender = time.time()
            curr_time = ender - starter
        else:
            ender = time.time()
            curr_time = ender - starter

        if math.isnan(score):
            if score > 0:
                score = 1e8
            else:
                score = -1e8
        if math.isinf(score):
            if score > 0:
                score = 1e8
            else:
                score = -1e8

        return score, curr_time

    def evaluate_wrapper_origin(self, arch, device: str, space_name: str,
                                batch_data: torch.tensor,
                                batch_labels: torch.tensor) -> (float, float):
        """
        :param arch: architecture to be scored
        :param device: cpu or GPU
        :param space_name: search space name
        :param batch_data: a mini batch of data, [ batch_size, channel, W, H ]
        :param batch_labels: a mini batch of labels
        :return: score, timeUsage
        """

        arch.train()
        arch.zero_grad()

        # measure scoring time
        if "cuda" in device:
            torch.cuda.synchronize()
            # use this will not need cuda.sync
            # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            # starter.record()
            starter, ender = time.time(), time.time()
        else:
            starter, ender = time.time(), time.time()

        # score
        score = self.evaluate(arch, device, batch_data, batch_labels, space_name)

        if "cuda" in device:
            # ender.record()
            # implicitly waits for the event to be marked as complete before calculating the time difference
            # curr_time = starter.elapsed_time(ender)
            torch.cuda.synchronize()
            ender = time.time()
            curr_time = ender - starter
        else:
            ender = time.time()
            curr_time = ender - starter

        if math.isnan(score):
            if score > 0:
                score = 1e8
            else:
                score = -1e8
        if math.isinf(score):
            if score > 0:
                score = 1e8
            else:
                score = -1e8

        return score, curr_time
