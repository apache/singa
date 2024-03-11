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

from src.common.constant import CommonVars
from src.controller.sampler_ea.regularized_ea import RegularizedEASampler
from src.controller.sampler_rl.reinforcement_learning import RLSampler
from src.controller.sampler_all.seq_sampler import SequenceSampler

sampler_register = {
    CommonVars.TEST_SAMPLER: SequenceSampler,
    CommonVars.RANDOM_SAMPLER: SequenceSampler,
    CommonVars.RL_SAMPLER: RLSampler,
    CommonVars.EA_SAMPLER: RegularizedEASampler,
}

