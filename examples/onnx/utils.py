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
# under th

import os
import urllib.request
import tarfile
import glob
import onnx
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(message)s')


def download_model(url):
    download_dir = '/tmp/'
    with tarfile.open(check_exist_or_download(url), 'r') as t:
        t.extractall(path=download_dir)


def load_dataset(test_data_dir):
    # load inputs
    inputs = []
    inputs_num = len(glob.glob(os.path.join(test_data_dir, 'input_*.pb')))
    for i in range(inputs_num):
        input_file = os.path.join(test_data_dir, 'input_{}.pb'.format(i))
        onnx_tensor = onnx.TensorProto()
        with open(input_file, 'rb') as f:
            onnx_tensor.ParseFromString(f.read())
        inputs.append(onnx.numpy_helper.to_array(onnx_tensor))

    # load reference outputs
    ref_outputs = []
    ref_outputs_num = len(glob.glob(os.path.join(test_data_dir, 'output_*.pb')))
    for i in range(ref_outputs_num):
        output_file = os.path.join(test_data_dir, 'output_{}.pb'.format(i))
        onnx_tensor = onnx.TensorProto()
        with open(output_file, 'rb') as f:
            onnx_tensor.ParseFromString(f.read())
        ref_outputs.append(onnx.numpy_helper.to_array(onnx_tensor))
    return inputs, ref_outputs


def check_exist_or_download(url):
    download_dir = '/tmp/'
    name = url.rsplit('/', 1)[-1]
    filename = os.path.join(download_dir, name)
    if not os.path.isfile(filename):
        logging.info("Downloading %s" % url)
        urllib.request.urlretrieve(url, filename)
    return filename
