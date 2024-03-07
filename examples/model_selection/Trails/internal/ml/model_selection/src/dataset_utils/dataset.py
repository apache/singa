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

import torch
from torch import tensor
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.datasets import MNIST, CIFAR10, CIFAR100, SVHN
from torchvision.transforms import Compose
from torchvision import transforms
from .imagenet16 import *


def get_dataloader(train_batch_size: int, test_batch_size: int, dataset: str,
                   num_workers: int, datadir: str, resize=None) -> (DataLoader, DataLoader, int):
    """
    Load CIFAR or imagenet datasets
    :param train_batch_size:
    :param test_batch_size:
    :param dataset: ImageNet16, cifar, svhn, ImageNet1k, mnist
    :param num_workers:
    :param datadir:
    :param resize:
    :return:
    """

    class_num = 0
    mean = []
    std = []
    pad = 0

    if 'ImageNet16' in dataset:
        mean = [x / 255 for x in [122.68, 116.66, 104.01]]
        std = [x / 255 for x in [63.22, 61.26, 65.09]]
        size, pad = 16, 2
    elif 'cifar' in dataset:
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        size, pad = 32, 4
    elif 'svhn' in dataset:
        mean = (0.5, 0.5, 0.5)
        std = (0.5, 0.5, 0.5)
        size, pad = 32, 0
    elif dataset == 'ImageNet1k':
        from .h5py_dataset import H5Dataset
        size, pad = 224, 2
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        # resize = 256
    elif dataset == 'ImageNet224-120':
        from .h5py_dataset import H5Dataset
        size, pad = 224, 2
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        # resize = 256

    if resize is None:
        resize = size

    train_transform = transforms.Compose([
        transforms.RandomCrop(size, padding=pad),
        transforms.Resize(resize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    if dataset == 'cifar10':
        class_num = 10
        train_dataset = CIFAR10(datadir, True, train_transform, download=True)
        test_dataset = CIFAR10(datadir, False, test_transform, download=True)
    elif dataset == 'cifar100':
        class_num = 100
        train_dataset = CIFAR100(datadir, True, train_transform, download=True)
        test_dataset = CIFAR100(datadir, False, test_transform, download=True)
    elif dataset == 'svhn':
        class_num = 10
        train_dataset = SVHN(datadir, split='train', transform=train_transform, download=True)
        test_dataset = SVHN(datadir, split='test', transform=test_transform, download=True)
    elif dataset == 'ImageNet16-120':
        class_num = 120
        train_dataset = ImageNet16(os.path.join(datadir, 'ImageNet16'), True, train_transform, 120)
        test_dataset = ImageNet16(os.path.join(datadir, 'ImageNet16'), False, test_transform, 120)
    elif dataset == 'ImageNet1k':
        class_num = 1000
        # train_dataset = ImageFolder(root=os.path.join(datadir, 'imagenet/val'), transform=train_transform)
        test_dataset = ImageFolder(root=os.path.join(datadir, 'imagenet/val'), transform=test_transform)
        train_dataset = test_dataset
    elif dataset == 'ImageNet224-120':
        class_num = 120
        test_dataset = ImageFolder(root=os.path.join(datadir, 'imagenet/val'), transform=test_transform)

        # get 0-120 classes
        class_indices = list(range(120))  # 0-120 inclusive
        subset_indices = [i for i, (_, label) in enumerate(test_dataset.samples) if label in class_indices]
        filtered_test_dataset = Subset(test_dataset, subset_indices)
        # get 0-120 classes
        train_dataset = filtered_test_dataset
        test_dataset = filtered_test_dataset
    elif dataset == 'mnist':
        data_transform = Compose([transforms.ToTensor()])
        # Normalise? transforms.Normalize((0.1307,), (0.3081,))
        train_dataset = MNIST("_dataset", True, data_transform, download=True)
        test_dataset = MNIST("_dataset", False, data_transform, download=True)
    else:
        raise ValueError('There are no more cifars or imagenets.')

    train_loader = DataLoader(
        train_dataset,
        train_batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=False
    )
    test_loader = DataLoader(
        test_dataset,
        test_batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=False
    )

    print("dataset load done")

    return train_loader, test_loader, class_num


def get_mini_batch(dataloader: DataLoader, sample_alg: str, batch_size: int, num_classes: int) -> (tensor, tensor):
    """
    Get a mini-batch of data,
    :param dataloader: DataLoader
    :param sample_alg: random or grasp
    :param batch_size: batch_size
    :param num_classes: num_classes
    :return: two tensor
    """

    if sample_alg == 'random':
        inputs, targets = _get_some_data(dataloader, batch_size=batch_size)
    elif sample_alg == 'grasp':
        inputs, targets = _get_some_data_grasp(dataloader, num_classes, samples_per_class=batch_size // num_classes)
    else:
        raise NotImplementedError(f'dataload {sample_alg} is not supported')

    return inputs, targets


def _get_some_data(train_dataloader: DataLoader, batch_size: int) -> (torch.tensor, torch.tensor):
    """
    Randomly sample some data, some class may not be sampled
    :param train_dataloader: torch dataLoader
    :param batch_size: batch_size of the data.
    :return:
    """
    traindata = []

    dataloader_iter = iter(train_dataloader)
    traindata.append(next(dataloader_iter))

    inputs = torch.cat([a for a, _ in traindata])
    targets = torch.cat([b for _, b in traindata])
    inputs = inputs
    targets = targets
    return inputs, targets


def _get_some_data_grasp(train_dataloader: DataLoader, num_classes: int,
                         samples_per_class: int) -> (torch.tensor, torch.tensor):
    """
    Sample some data while guarantee example class has equal number of samples.
    :param train_dataloader: torch dataLoader
    :param num_classes: number of class
    :param samples_per_class:  how many samples for eacl class.
    :return:
    """

    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    dataloader_iter = iter(train_dataloader)
    while True:
        inputs, targets = next(dataloader_iter)
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx:idx + 1], targets[idx:idx + 1]
            category = y.item()
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break

    x = torch.cat([torch.cat(_, 0) for _ in datas])
    y = torch.cat([torch.cat(_) for _ in labels]).view(-1)
    return x, y
