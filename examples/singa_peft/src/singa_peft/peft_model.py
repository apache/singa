#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#

from singa import model
from singa_peft.peft_config import PeftConfig
from singa_peft.peft_registry import PeftRegistry


class PeftModel(model.Model):
    """
    PeftModel: modify the base model based on the peft config. A Wrapper of model and tuner.
    """
    def __init__(self, base_model: model.Model, peft_config: PeftConfig):
        r"""
        Args:
            base_model: the base model
            peft_config: the config of peft
        """
        super().__init__()
        self.base_model = base_model
        self.peft_config = peft_config
        self.peft_type = peft_config.peft_type
        self.dimension = self.base_model.dimension
        # Get the injected tuner class based on peft_type
        cls = PeftRegistry.get_tuner(self.peft_type)
        self.tuner = cls(peft_config)
        # Inject adapter into base_model
        self.base_model = self.tuner.inject(base_model)

    def forward(self, inputs):
        return self.base_model.forward(inputs)

    def train_one_batch(self, x, y, dist_option, spars):
        return self.base_model.train_one_batch(x, y, dist_option, spars)

    def set_optimizer(self, optimizer):
        self.base_model.set_optimizer(optimizer)

    def compile(self, inputs, is_train=True, use_graph=False, sequential=False):
        self.base_model.compile(inputs, is_train, use_graph, sequential)

    def train(self, mode=True):
        super().train(mode)
        self.base_model.train(mode)

    def eval(self):
        super().eval()
        self.base_model.eval()

    def merge_weights(self, mode=True):
        self.tuner.merge_weights(self.base_model, mode)

    def get_params(self):
        params =  self.base_model.get_params()
        return params

    def set_params(self, params):
        self.base_model.set_params(params)


def get_peft_model(base_model: model.Model, peft_config: PeftConfig):
    r"""
    Args:
        base_model: the base model
        peft_config: the config of peft

    Returns: a peft model based on peft config
    """
    peft_model = PeftModel(base_model, peft_config)
    return peft_model
