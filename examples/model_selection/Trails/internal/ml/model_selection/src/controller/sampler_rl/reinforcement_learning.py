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


class ExponentialMovingAverage(object):
    """Class that maintains an exponential moving average."""

    def __init__(self, momentum):
        self._numerator = 0
        self._denominator = 0
        self._momentum = momentum

    def update(self, value):
        self._numerator = (
            self._momentum * self._denominator + (1 - self._momentum) * value
        )
        self._denominator = self._momentum * self._denominator + (1 - self._momentum)

    def value(self):
        """Return the current value of the moving average"""
        return self._numerator / self._denominator


class RLSampler(Sampler):

    def __init__(self, space: SpaceWrapper, args):

        super().__init__(space)

        self.policy = self.space.get_reinforcement_learning_policy(args.rl_learning_rate)
        # update policy's parameters
        self.baseline = ExponentialMovingAverage(args.rl_EMA_momentum)
        self.log_prob = 0

    def sample_next_arch(self, max_nodes: int) -> (str, ModelMicroCfg):
        while True:
            self.log_prob, action = self.policy.select_action()
            arch_struct = self.policy.generate_arch(action)
            arch_id = self.space.arch_to_id(arch_struct)
            yield arch_id, arch_struct

    def fit_sampler(self, score: float):
        reward = score
        self.baseline.update(reward)
        self.policy.update_policy(reward, self.baseline.value(), self.log_prob)
