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


import collections
from src.search_space.core.model_params import ModelMicroCfg
from src.controller.core.sample import Sampler
import random
from src.search_space.core.space import SpaceWrapper


class Model(object):
    def __init__(self):
        self.arch = None
        self.score = None

    def __str__(self):
        """Prints a readable version of this bitstring."""
        return "{:}".format(self.arch)


class RegularizedEASampler(Sampler):

    def __init__(self, space: SpaceWrapper, population_size: int, sample_size: int):
        super().__init__(space)

        self.population_size = population_size
        # list of object,
        self.population = collections.deque()
        # list of str, for duplicate checking
        self.population_model_ids = collections.deque()

        self.space = space
        self.sample_size = sample_size
        self.current_sampled = 0

        # id here is to match the outside value.
        self.current_arch_id = None
        self.current_arch_micro = None

        # use the visited to reduce the collapse
        self.visited = {}
        self.max_mutate_time = 4
        self.max_mutate_sampler_time = 4

    def sample_next_arch(self, sorted_model_ids: list) -> (str, ModelMicroCfg):
        """
        This function performs one evolution cycle. It produces a model and removes another.
        Models are sampled randomly from the current population. If the population size is less than the
        desired population size, a random architecture is added to the population.

        :param sorted_model_ids: List of model ids sorted based on some criterion (not used here directly).
        :return: Tuple of the architecture id and the architecture configuration (micro).
        """
        # Case 1: If population hasn't reached desired size, add random architectures
        if len(self.population) < self.population_size:
            while True:
                arch_id, arch_micro = self.space.random_architecture_id()
                # Ensure that EA population has no repeated value
                if str(arch_id) not in self.population_model_ids:
                    break
            self.current_arch_micro = arch_micro
            self.current_arch_id = arch_id
            return arch_id, arch_micro

        # Case 2: If population has reached desired size, evolve population
        else:
            cur_mutate_sampler_time = 0
            is_found_new = False

            # Keep attempting mutations for a maximum of 'max_mutate_sampler_time' times
            while cur_mutate_sampler_time < self.max_mutate_sampler_time:
                cur_mutate_time = 0

                # Randomly select a sample of models from the population
                sample = []
                sample_ids = []
                while len(sample) < self.sample_size:
                    candidate = random.choice(list(self.population))
                    candidate_id = self.population_model_ids[self.population.index(candidate)]
                    sample.append(candidate)
                    sample_ids.append(candidate_id)

                # Select the best parent from the sample (based on the order in sorted_model_ids)
                parent_id = max(sample_ids, key=lambda _id: sorted_model_ids.index(str(_id)))
                parent = sample[sample_ids.index(parent_id)]

                # Try to mutate the parent up to 'max_mutate_time' times
                while cur_mutate_time < self.max_mutate_time:
                    arch_id, arch_micro = self.space.mutate_architecture(parent.arch)

                    # If the mutated architecture hasn't been visited or we've visited all possible architectures, stop
                    if arch_id not in self.visited or len(self.space) == len(self.visited):
                        self.visited[arch_id] = True
                        is_found_new = True
                        break
                    cur_mutate_time += 1

                # If we've found a new architecture, stop sampling
                if is_found_new:
                    break

                cur_mutate_sampler_time += 1

            # If we've hit the maximum number of mutation attempts, do nothing
            if cur_mutate_time * cur_mutate_sampler_time == self.max_mutate_time * self.max_mutate_sampler_time:
                pass

            # Update current architecture details
            self.current_arch_micro = arch_micro
            self.current_arch_id = arch_id

            return arch_id, arch_micro

    def fit_sampler(self, score: float):
        # if it's in Initialize stage, add to the population with random models.
        if len(self.population) < self.population_size:
            model = Model()
            model.arch = self.current_arch_micro
            model.score = score
            self.population.append(model)
            self.population_model_ids.append(self.current_arch_id)

        # if it's in mutation stage
        else:
            child = Model()
            child.arch = self.current_arch_micro
            child.score = score

            self.population.append(child)
            self.population_model_ids.append(self.current_arch_id)
            # Remove the oldest model.
            self.population.popleft()
            self.population_model_ids.popleft()


class AsyncRegularizedEASampler(Sampler):

    def __init__(self, space: SpaceWrapper, population_size: int, sample_size: int):
        super().__init__(space)

        self.population_size = population_size
        # list of object,
        self.population = collections.deque()
        # list of str, for duplicate checking
        self.population_model_ids = collections.deque()

        self.space = space
        self.sample_size = sample_size
        self.current_sampled = 0

        # use the visited to reduce the collapse
        self.visited = {}
        self.max_mutate_time = 2
        self.max_mutate_sampler_time = 3

    def sample_next_arch(self, sorted_model_ids: list) -> (str, ModelMicroCfg):
        # Case 1: If population hasn't reached desired size, add random architectures
        if len(self.population) < self.population_size:
            while True:
                arch_id, arch_micro = self.space.random_architecture_id()
                # Ensure that EA population has no repeated value
                if str(arch_id) not in self.population_model_ids:
                    break
            return arch_id, arch_micro

        # Case 2: If population has reached desired size, evolve population
        else:
            cur_mutate_sampler_time = 0
            is_found_new = False

            # Keep attempting mutations for a maximum of 'max_mutate_sampler_time' times
            while cur_mutate_sampler_time < self.max_mutate_sampler_time:
                cur_mutate_time = 0

                # Randomly select a sample of models from the population
                sample = []
                sample_ids = []
                while len(sample) < self.sample_size:
                    candidate = random.choice(list(self.population))
                    candidate_id = self.population_model_ids[self.population.index(candidate)]
                    sample.append(candidate)
                    sample_ids.append(candidate_id)

                # Select the best parent from the sample (based on the order in sorted_model_ids)
                parent_id = max(sample_ids, key=lambda _id: sorted_model_ids.index(str(_id)))
                parent = sample[sample_ids.index(parent_id)]

                # Try to mutate the parent up to 'max_mutate_time' times
                while cur_mutate_time < self.max_mutate_time:
                    arch_id, arch_micro = self.space.mutate_architecture(parent.arch)

                    # If the mutated architecture hasn't been visited or we've visited all possible architectures, stop
                    if arch_id not in self.visited or len(self.space) == len(self.visited):
                        self.visited[arch_id] = True
                        is_found_new = True
                        break
                    cur_mutate_time += 1

                # If we've found a new architecture, stop sampling
                if is_found_new:
                    break

                cur_mutate_sampler_time += 1

            # If we've hit the maximum number of mutation attempts, do nothing
            if cur_mutate_time * cur_mutate_sampler_time == self.max_mutate_time * self.max_mutate_sampler_time:
                pass

            # Update current architecture details
            return arch_id, arch_micro

    def async_fit_sampler(self, current_arch_id, current_arch_micro, score: float):
        # if it's in Initialize stage, add to the population with random models.
        if len(self.population) < self.population_size:
            model = Model()
            model.arch = current_arch_micro
            model.score = score
            self.population.append(model)
            self.population_model_ids.append(current_arch_id)

        # if it's in mutation stage
        else:
            child = Model()
            child.arch = current_arch_micro
            child.score = score

            self.population.append(child)
            self.population_model_ids.append(current_arch_id)
            # Remove the oldest model.
            self.population.popleft()
            self.population_model_ids.popleft()
