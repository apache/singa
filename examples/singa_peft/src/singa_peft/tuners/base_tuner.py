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

from abc import ABC, abstractmethod
from singa import model


class BaseTuner(ABC):
    """
    BaseTuner: the base class of all tuner，all PEFT methods must inherit this class and implement the inject method.
    """
    def __init__(self, config):
        r"""
        Args:
            config: object of the PeftConfig class or its subclasses
        """
        self.config = config

    @abstractmethod
    def inject(self, base_model: model.Model) -> model.Model:
        r"""
        all PEFT methods must implement the inject method, inject the peft method into the base model.
        Args:
            base_model: the base model

        Returns: the base model with inject method
        """
        raise NotImplementedError

    @abstractmethod
    def merge_weights(self, base_model: model.Model, mode: bool = True) -> model.Model:
        r"""
        all PEFT methods must implement the merge_weights method. After model training, weights need to be combined to speed up inference
        Args:
            base_model: the base model with inject method
            mode: merge parameters or not, default True

        Returns: the model with inject method after combining weights
        """
        raise NotImplementedError

    @staticmethod
    def freeze_base_parameters(base_model: model.Model):
        r"""
        freeze the weights of the base model
        Args:
            base_model: the base model
        """
        params = base_model.get_params()
        for k, v in params.items():
            v.requires_grad = False
            v.stores_grad = False


