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


import h5py
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader


class H5Dataset(Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_path = h5_path
        self.h5_file = None
        self.length = len(h5py.File(h5_path, 'r'))
        self.transform = transform

    def __getitem__(self, index):

        # loading in getitem allows us to use multiple processes for data loading
        # because hdf5 files aren't pickelable so can't transfer them across processes
        # https://discuss.pytorch.org/t/hdf5-a-data-format-for-pytorch/40379
        # https://discuss.pytorch.org/t/dataloader-when-num-worker-0-there-is-bug/25643/16
        # TODO possible look at __getstate__ and __setstate__ as a more elegant solution
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_path, 'r')

        record = self.h5_file[str(index)]

        if self.transform:
            x = Image.fromarray(record['data'][()])
            x = self.transform(x)
        else:
            x = torch.from_numpy(record['data'][()])

        y = record['target'][()]
        y = torch.from_numpy(np.asarray(y))

        return (x, y)

    def __len__(self):
        return self.length
