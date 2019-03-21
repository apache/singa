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


# load and run the onnx model exported from pytorch
# https://github.com/onnx/tutorials/blob/master/tutorials/PytorchOnnxExport.ipynb


import argparse
from singa import device
from singa import sonnx
from singa import tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load model from pytorch")
    parser.add_argument("--use_cpu", action="store_true")
    args = parser.parse_args()
    if args.use_cpu:
        print("Using CPU")
        dev = device.get_default_device()
    else:
        print("Using GPU")
        dev = device.create_cuda_gpu()
    model = sonnx.load("alexnet.onnx")
    backend = sonnx.prepare(model, dev)
    input_name = model.graph.inputs[0].name
    inputs = tensor.Tensor(shape=(2, 3, 224, 224), device=dev, name=input_name)
    inputs.gaussian(0, 0.01)
    y = backend.run([inputs])[0]
