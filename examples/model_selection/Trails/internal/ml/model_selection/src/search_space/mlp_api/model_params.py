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


from src.search_space.core.model_params import ModelMacroCfg


class MlpMacroCfg(ModelMacroCfg):

    def __init__(self, nfield: int, nfeat: int, nemb: int,
                 num_layers: int,
                 num_labels: int,
                 layer_choices: list):
        super().__init__(num_labels)

        self.nfield = nfield
        self.nfeat = nfeat
        self.nemb = nemb
        self.layer_choices = layer_choices
        self.num_layers = num_layers
