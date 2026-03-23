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

from singa import tensor
from singa import device
from singa import opt
import numpy as np
from singa_peft import get_peft_model
from singa_peft import LinearLoraConfig
from examples.model.mlp import MLP

np.random.seed(0)
np_dtype = {"float16": np.float16, "float32": np.float32}
singa_dtype = {"float16": tensor.float16, "float32": tensor.float32}


if __name__ == '__main__':
    f = lambda x: (5 * x + 1)
    bd_x = np.linspace(-1.0, 1, 200)
    bd_y = f(bd_x)

    # choose one precision
    precision = singa_dtype["float32"]
    np_precision = np_dtype["float32"]

    dev = device.get_default_device()
    sgd = opt.SGD(0.5, 0.9, 1e-5, dtype=singa_dtype["float32"])
    tx = tensor.Tensor((400, 2), dev, precision)
    ty = tensor.Tensor((400,), dev, tensor.int32)
    model = MLP(in_features=2, perceptron_size=3, num_classes=2)
    model.set_optimizer(sgd)
    model.compile([tx], is_train=True, use_graph=False, sequential=True)
    model.train()
    print("-----0-----")
    print(model.get_params())
    for i in range(10):
        # generate the training data
        x = np.random.uniform(-1, 1, 400)
        y = f(x) + 2 * np.random.randn(len(x))
        # convert training data to 2d space
        label = np.asarray([5 * a + 1 > b for (a, b) in zip(x, y)]).astype(np.int32)
        data = np.array([[a, b] for (a, b) in zip(x, y)], dtype=np_precision)
        tx.copy_from_numpy(data)
        ty.copy_from_numpy(label)
        out, loss = model(tx, ty, 'plain', spars=None)
        print("training loss = ", tensor.to_numpy(loss)[0])
    print("-----1-----")
    print(model.get_params())

    config = LinearLoraConfig(4, 1, 0.2, ["linear1", "linear2"])

    peft_model = get_peft_model(model, config)
    peft_model.set_optimizer(sgd)
    peft_model.compile([tx], is_train=True, use_graph=False, sequential=True)
    peft_model.train()
    print("-----2-----")
    print(peft_model.get_params())

    for i in range(10):
        # generate the training data
        x = np.random.uniform(-1, 1, 400)
        y = f(x) + 2 * np.random.randn(len(x))
        # convert training data to 2d space
        label = np.asarray([5 * a + 1 > b for (a, b) in zip(x, y)]).astype(np.int32)
        data = np.array([[a, b] for (a, b) in zip(x, y)], dtype=np_precision)
        tx.copy_from_numpy(data)
        ty.copy_from_numpy(label)
        out, loss = peft_model(tx, ty, 'plain', spars=None)
        print("training loss = ", tensor.to_numpy(loss)[0])
    print("-----3-----")
    print(peft_model.get_params())
  
