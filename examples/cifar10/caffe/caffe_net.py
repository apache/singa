from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import zip
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
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
import urllib.request, urllib.parse, urllib.error
from singa import converter


def create_net(use_cpu):
    urllib.request.urlretrieve("https://raw.githubusercontent.com/BVLC/caffe/master/examples/cifar10/cifar10_full_train_test.prototxt", "train_test.prototxt")
    urllib.request.urlretrieve("https://raw.githubusercontent.com/BVLC/caffe/master/examples/cifar10/cifar10_full_solver.prototxt", "solver.prototxt")
    input_sample_shape = [3, 32, 32, ]

    cvt = converter.CaffeConverter("train_test.prototxt", "solver.prototxt",
                                   input_sample_shape)
    net = cvt.create_net()
    for (p, specs) in zip(net.param_values(), net.param_specs()):
        filler = specs.filler
        if filler.type == 'gaussian':
            p.gaussian(filler.mean, filler.std)
        else:
            p.set_value(0)
        print(specs.name, filler.type, p.l1())

    return net
