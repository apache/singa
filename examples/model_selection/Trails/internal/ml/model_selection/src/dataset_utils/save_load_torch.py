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


import argparse

from tqdm import tqdm
import torch
import os
import glob


def decode_libsvm(line):
    columns = line.split(' ')
    map_func = lambda pair: (int(pair[0]), float(pair[1]))
    id, value = zip(*map(lambda col: map_func(col.split(':')), columns[1:]))
    sample = {'id': torch.LongTensor(id),
              'value': torch.FloatTensor(value),
              'y': float(columns[0])}
    return sample


def _save_data(data_dir, fname, nfields, namespace):
    with open(fname) as f:
        sample_lines = sum(1 for line in f)

    feat_id = torch.LongTensor(sample_lines, nfields)
    feat_value = torch.FloatTensor(sample_lines, nfields)
    y = torch.FloatTensor(sample_lines)

    nsamples = 0
    with tqdm(total=sample_lines) as pbar:
        with open(fname) as fp:
            line = fp.readline()
            while line:
                try:
                    sample = decode_libsvm(line)
                    feat_id[nsamples] = sample['id']
                    feat_value[nsamples] = sample['value']
                    y[nsamples] = sample['y']
                    nsamples += 1
                except Exception:
                    print(f'incorrect data format line "{line}" !')
                line = fp.readline()
                pbar.update(1)
    print(f'# {nsamples} data samples loaded...')

    # save the tensors to disk
    torch.save(feat_id, f'{data_dir}/{namespace}_feat_id.pt')
    torch.save(feat_value, f'{data_dir}/{namespace}_feat_value.pt')
    torch.save(y, f'{data_dir}/{namespace}_y.pt')


def parse_arguments():
    parser = argparse.ArgumentParser(description='FastAutoNAS')

    parser.add_argument('--nfield', type=int, default=10,
                        help='the number of fields, frappe: 10, uci_diabetes: 43, criteo: 39')

    parser.add_argument('--dataset', type=str, default='frappe',
                        help='cifar10, cifar100, ImageNet16-120, frappe, criteo, uci_diabetes')

    return parser.parse_args()


def load_data(data_dir, namespace):
    feat_id = torch.load(f'{data_dir}/{namespace}_feat_id.pt')
    feat_value = torch.load(f'{data_dir}/{namespace}_feat_value.pt')
    y = torch.load(f'{data_dir}/{namespace}_y.pt')

    print(f'# {int(y.shape[0])} data samples loaded...')

    return feat_id, feat_value, y, int(y.shape[0])


if __name__ == "__main__":
    args = parse_arguments()

    _data_dir = os.path.join("./dataset", args.dataset)

    train_name_space = "decoded_train"
    valid_name_space = "decoded_valid"
    # save
    train_file = glob.glob("%s/tr*libsvm" % _data_dir)[0]
    val_file = glob.glob("%s/va*libsvm" % _data_dir)[0]
    _save_data(_data_dir, train_file, args.nfield, train_name_space)
    _save_data(_data_dir, val_file, args.nfield, valid_name_space)

    # read
    # load_data(data_dir, train_name_space)
    # load_data(data_dir, valid_name_space)
