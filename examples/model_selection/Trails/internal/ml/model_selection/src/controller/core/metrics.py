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


from enum import Enum, auto


class Metric(Enum):
    RAW = auto()
    ALL = auto()

    TRAIN_ACCURACY = auto()
    VAL_ACCURACY = auto()
    TEST_ACCURACY = auto()

    TRAIN_LOSS = auto()
    VAL_LOSS = auto()
    TEST_LOSS = auto()

    TRAIN_TIME = auto()
    VAL_TIME = auto()
    TEST_TIME = auto()

    FLOPS = auto()
    LATENCY = auto()
    PARAMETERS = auto()
    EPOCH = auto()
    HP = auto()




