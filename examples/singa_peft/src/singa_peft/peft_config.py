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

class PeftConfig:
    """
    PeftConfig: the base class of all PEFT methods config, defines the common configuration parameters for all PEFT methods.
    """
    def __init__(self, peft_type: str, freeze_base_model: bool = True):
        r"""
        peft_type: the type of peft, linear_lora etc...
        freeze_base_model: freeze base model parameters, default true
        Args:
            peft_type:
            freeze_base_model:
        """
        self.peft_type = peft_type
        self.freeze_base_model = freeze_base_model
