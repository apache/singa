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

from singa import model, layer
from singa_peft.peft_registry import PeftRegistry
from singa_peft.tuners.base_tuner import BaseTuner
from singa_peft.tuners.linear_lora.config import LinearLoraConfig
from singa_peft.tuners.linear_lora.layer import LinearLoRALayer


@PeftRegistry.register("linear_lora")
class LinearLoraTuner(BaseTuner):

    def __init__(self, config):
        super().__init__(config)
        self.targeted_layers = []

    def inject(self, base_model: model.Model) -> model.Model:
        # freeze base_model parameters
        if self.config.freeze_base_model:
            self.freeze_base_parameters(base_model)
        return self._inject_linear_lora(base_model, self.config)

    def _inject_linear_lora(self, base_model, config: LinearLoraConfig) -> model.Model:
        target_layers = config.target_layers
        r = config.r
        lora_alpha = config.lora_alpha
        lora_dropout = config.lora_dropout
        for target_layer in target_layers:
            base_layer = getattr(base_model, target_layer)
            if base_layer is not None and isinstance(base_layer, layer.Linear):
                self.targeted_layers.append(target_layer)
                new_layer = LinearLoRALayer(base_layer, r, lora_alpha, lora_dropout)
                setattr(base_model, target_layer, new_layer)
        return base_model

    def merge_weights(self, base_model: model.Model, mode: bool = True) -> model.Model:
        for target_layer in self.targeted_layers:
            base_layer = getattr(base_model, target_layer)
            if base_layer is not None:
                base_layer.merge_weights(mode)
        return base_model