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


import os
from src.common.constant import Config
from src.search_space.core.space import SpaceWrapper
from src.query_api.query_api_img import ImgScoreQueryApi


def init_search_space(args) -> SpaceWrapper:
    """
    :param args:
    :param loapi: Local score API, records all scored arch, 101 use it to detect which arch is scored.
    :return:
    """

    if args.search_space == Config.MLPSP:
        from .mlp_api.space import MlpSpace
        from .mlp_api.model_params import MlpMacroCfg
        from .mlp_api.space import DEFAULT_LAYER_CHOICES_20, DEFAULT_LAYER_CHOICES_10
        print("[Singa] src/search_space/init_search_space.py config.MLPSP")
        if args.hidden_choice_len == 10:
            model_cfg = MlpMacroCfg(
                args.nfield,
                args.nfeat,
                args.nemb,
                args.num_layers,
                args.num_labels,
                DEFAULT_LAYER_CHOICES_10)
        else:
            model_cfg = MlpMacroCfg(
                args.nfield,
                args.nfeat,
                args.nemb,
                args.num_layers,
                args.num_labels,
                DEFAULT_LAYER_CHOICES_20)

        return MlpSpace(model_cfg)
    else:
        raise Exception
