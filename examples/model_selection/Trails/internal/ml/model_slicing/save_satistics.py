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


from main import parse_arguments, seed_everything
import os
import glob
import json
from model_slicing.algorithm.src.data_loader import SQLAttacedLibsvmDataset


def write_json(file_name, data):
    print(f"writting {file_name}...")
    with open(file_name, 'w') as outfile:
        outfile.write(json.dumps(data))


args = parse_arguments()
seed_everything(args.seed)


data_dir = os.path.join(args.data_dir, args.dataset)
train_file = glob.glob("%s/tr*libsvm" % data_dir)[0]


train_loader = SQLAttacedLibsvmDataset(
    train_file,
    args.nfield,
    args.max_filter_col)


write_json(
    f"{args.dataset}_col_cardinalities",
    train_loader.col_cardinalities)

