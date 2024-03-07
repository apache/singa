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


import random

from src.controller.core.sample import Sampler
from src.search_space.core.model_params import ModelMicroCfg
from src.search_space.core.space import SpaceWrapper


class SequenceSampler(Sampler):

    def __init__(self, space: SpaceWrapper):
        super().__init__(space)

        self.arch_gene = self.space.sample_all_models()

    def sample_next_arch(self, sorted_model: list = None) -> (str, ModelMicroCfg):
        """
        Sample one random architecture, can sample max 10k architectures.
        :return: arch_id, architecture
        """

        try:
            arch_id, arch_micro = self.arch_gene.__next__()
            return arch_id, arch_micro
        except Exception as e:
            if "StopIteration" in str(e):
                print("the end")
                return None, None
            else:
                print("Error", str(e))
                return None, None

    def fit_sampler(self, score: float):
        pass
