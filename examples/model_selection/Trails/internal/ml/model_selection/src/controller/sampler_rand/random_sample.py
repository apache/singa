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


from src.controller.core.sample import Sampler
from src.search_space.core.space import SpaceWrapper
from src.search_space.core.model_params import ModelMicroCfg


class RandomSampler(Sampler):

    def __init__(self, space: SpaceWrapper):
        super().__init__(space)
        self.visited = []

    def sample_next_arch(self, sorted_model: list = None) -> (str, ModelMicroCfg):
        while True:
            arch_id, model_micro = self.space.random_architecture_id()

            if arch_id not in self.visited:
                self.visited.append(arch_id)
                return str(arch_id), model_micro

    def fit_sampler(self, score: float):
        # random sampler can skip this.
        pass
