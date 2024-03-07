##################################################
# Copyright (c) Xuanyi Dong [GitHub D-X-Y], 2019 #
##################################################

import os, sys, hashlib
import numpy as np
from PIL import Image
import torch.utils.data as data

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


def calculate_md5(fpath, chunk_size=1024 * 1024):
    md5 = hashlib.md5()
    with open(fpath, 'rb') as f:
        for chunk in iter(lambda: f.read(chunk_size), b''):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath, md5, **kwargs):
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath, md5=None):
    if not os.path.isfile(fpath): return False
    if md5 is None:
        return True
    else:
        return check_md5(fpath, md5)


class ImageNet16(data.Dataset):
    # http://image-net.org/download-images
    # A Downsampled Variant of ImageNet as an Alternative to the CIFAR datasets
    # https://arxiv.org/pdf/1707.08819.pdf

    train_list = [
        ['train_data_batch_1', '27846dcaa50de8e21a7d1a35f30f0e91'],
        ['train_data_batch_2', 'c7254a054e0e795c69120a5727050e3f'],
        ['train_data_batch_3', '4333d3df2e5ffb114b05d2ffc19b1e87'],
        ['train_data_batch_4', '1620cdf193304f4a92677b695d70d10f'],
        ['train_data_batch_5', '348b3c2fdbb3940c4e9e834affd3b18d'],
        ['train_data_batch_6', '6e765307c242a1b3d7d5ef9139b48945'],
        ['train_data_batch_7', '564926d8cbf8fc4818ba23d2faac7564'],
        ['train_data_batch_8', 'f4755871f718ccb653440b9dd0ebac66'],
        ['train_data_batch_9', 'bb6dd660c38c58552125b1a92f86b5d4'],
        ['train_data_batch_10', '8f03f34ac4b42271a294f91bf480f29b'],
    ]
    valid_list = [
        ['val_data', '3410e3017fdaefba8d5073aaa65e4bd6'],
    ]

    def __init__(self, root, train, transform, use_num_of_class_only=None):
        self.root = root
        self.transform = transform
        self.train = train  # training set or valid set
        if not self._check_integrity(): raise RuntimeError('Dataset not found or corrupted.')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.valid_list
        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for i, (file_name, checksum) in enumerate(downloaded_list):
            file_path = os.path.join(self.root, file_name)
            # print ('Load {:}/{:02d}-th : {:}'.format(i, len(downloaded_list), file_path))
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                self.targets.extend(entry['labels'])
        self.data = np.vstack(self.data).reshape(-1, 3, 16, 16)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        if use_num_of_class_only is not None:
            assert isinstance(use_num_of_class_only,
                              int) and use_num_of_class_only > 0 and use_num_of_class_only < 1000, 'invalid use_num_of_class_only : {:}'.format(
                use_num_of_class_only)
            new_data, new_targets = [], []
            for I, L in zip(self.data, self.targets):
                if 1 <= L <= use_num_of_class_only:
                    new_data.append(I)
                    new_targets.append(L)
            self.data = new_data
            self.targets = new_targets
        #    self.mean.append(entry['mean'])
        # self.mean = np.vstack(self.mean).reshape(-1, 3, 16, 16)
        # self.mean = np.mean(np.mean(np.mean(self.mean, axis=0), axis=1), axis=1)
        # print ('Mean : {:}'.format(self.mean))
        # temp      = self.data - np.reshape(self.mean, (1, 1, 1, 3))
        # std_data  = np.std(temp, axis=0)
        # std_data  = np.mean(np.mean(std_data, axis=0), axis=0)
        # print ('Std  : {:}'.format(std_data))

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index] - 1

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.valid_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, filename)
            if not check_integrity(fpath, md5):
                return False
        return True


if __name__ == '__main__':
    train = ImageNet16('/data02/dongxuanyi/.torch/cifar.python/ImageNet16', True, None)
    valid = ImageNet16('/data02/dongxuanyi/.torch/cifar.python/ImageNet16', False, None)

    print(len(train))
    print(len(valid))
    image, label = train[111]
    trainX = ImageNet16('/data02/dongxuanyi/.torch/cifar.python/ImageNet16', True, None, 200)
    validX = ImageNet16('/data02/dongxuanyi/.torch/cifar.python/ImageNet16', False, None, 200)
    print(len(trainX))
    print(len(validX))
    # import pdb; pdb.set_trace()
