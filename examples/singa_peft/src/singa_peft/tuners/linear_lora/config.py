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

from typing import Optional
from singa_peft.peft_config import PeftConfig


class LinearLoraConfig(PeftConfig):
    """
    LinearLoraConfig: linear lora config class
    """
    def __init__(self, r: int = 8, lora_alpha: int = 1, lora_dropout: float = 0, target_layers: Optional[list[str]] = None):
        r"""
        Args:
            r: the rank in LoRA, which determines the size of the low-rank matrix, default 8
            lora_alpha: learning rate scaling factor, default 1
            lora_dropout: dropout ratio, default 0.
            target_layers: list of the layer names to replace with LoRA. For examples, ['linear1', 'linear2']
        """
        super().__init__(peft_type="linear_lora")
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_layers = target_layers
