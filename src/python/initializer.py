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
# =============================================================================
"""Popular initialization methods for parameter values (Tensor ojects)"""

import math


def uniform(t, low=0, high=1):
    t.uniform(low, high)


def gaussian(t, mean=0, std=0.01):
    t.gaussian(mean, std)


def xavier(t):
    scale = math.sqrt(6.0 / (t.shape[0] + t.shape[1]))
    t.uniform(-scale, scale)


def msra(t):
    t.gaussian(0, math.sqrt(2.0 / t.shape[0]))
