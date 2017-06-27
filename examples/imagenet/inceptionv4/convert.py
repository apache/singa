# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple script for inspect checkpoint files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import cPickle as pickle
import os

import numpy as np
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import app
import model


FLAGS = None


def rename(name, suffix):
    p = name.rfind('/')
    if p == -1:
        print('Bad name=%s' % name)
    return name[0:p+1] + suffix


def convert(file_name):
    net, _ = model.create_net()
    params = {'SINGA_VERSION': 1101}
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        for pname, pval in zip(net.param_names(), net.param_values()):
            if 'weight' in pname:
                val = reader.get_tensor(rename(pname, 'weights'))
                if 'Conv' in pname:
                    val = val.transpose((3, 2, 0, 1))
                    val = val.reshape((val.shape[0], -1))
            elif 'bias' in pname:
                val = reader.get_tensor(rename(pname, 'biases'))
            elif 'mean' in pname:
                val = reader.get_tensor(rename(pname, 'moving_mean'))
            elif 'var' in pname:
                val = reader.get_tensor(rename(pname, 'moving_variance'))
            elif 'beta' in pname:
                val= reader.get_tensor(pname)
            elif 'gamma' in pname:
                val = np.ones(pval.shape)
            else:
                print('not matched param %s' % pname)
            assert val.shape == pval.shape, ('the shapes not match ', val.shape, pval.shape)
            params[pname] = val.astype(np.float32)
            print('converting:', pname, pval.shape)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in var_to_shape_map:
            if 'weights' in key:
                key = rename(key, 'weight')
            elif 'biases' in key:
                key = rename(key, 'bias')
            elif 'moving_mean' in key:
                key = rename(key, 'mean')
            elif 'moving_variance' in key:
                key = rename(key, 'var')
            if key not in params:
                print('key=%s not in the net' % key)
        '''
        for key in var_to_shape_map:
            print("tensor_name: ", key, var_to_shape_map[key])
        '''
        with open(os.path.splitext(file_name)[0] + '.pickle', 'wb') as fd:
            pickle.dump(params, fd)
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                    "with SNAPPY.")
        if ("Data loss" in str(e) and
            (any([e in file_name for e in [".index", ".meta", ".data"]]))):
            proposed_file = ".".join(file_name.split(".")[0:-1])
            v2_file_error_template = """
    It's likely that this is a V2 checkpoint and you need to provide the filename
    *prefix*.  Try removing the '.' and extension.  Try:
    inspect checkpoint --file_name = {}"""
        print(v2_file_error_template.format(proposed_file))



def main(unused_argv):
    if not FLAGS.file_name:
        print("Usage: convert.py --file_name=checkpoint_file_name ")
        sys.exit(1)
    else:
        convert(FLAGS.file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--file_name", type=str, default="", help="Checkpoint filename. "
                        "Note, if using Checkpoint V2 format, file_name is the "
                        "shared prefix between all files in the checkpoint.")
    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)
