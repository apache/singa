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
