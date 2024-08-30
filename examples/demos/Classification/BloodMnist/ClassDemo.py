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

import json
import os
import time
from glob import glob

import numpy as np
from PIL import Image
from singa import device, layer, model, opt, tensor
from tqdm import tqdm

from transforms import Compose, Normalize, ToTensor

np_dtype = {"float16": np.float16, "float32": np.float32}
singa_dtype = {"float16": tensor.float16, "float32": tensor.float32}


class ClassDataset(object):
    """Fetch data from file and generate batches.

    Load data from folder as PIL.Images and convert them into batch array.

    Args:
        img_folder (Str): Folder path of the training/validation images.
        transforms (Transform):  Preprocess transforms.
    """

    def __init__(self, img_folder, transforms):
        super(ClassDataset, self).__init__()

        self.img_list = list()
        self.transforms = transforms

        classes = os.listdir(img_folder)
        for i in classes:
            images = glob(os.path.join(img_folder, i, "*"))
            for img in images:
                self.img_list.append((img, i))

    def __len__(self) -> int:
        return len(self.img_list)

    def __getitem__(self, index: int):
        img_path, label_str = self.img_list[index]
        img = Image.open(img_path)
        img = self.transforms.forward(img)
        label = np.array(label_str, dtype=np.int32)

        return img, label

    def batchgenerator(self, indexes, batch_size, data_size):
        """Generate batch arrays from transformed image list.

        Args:
            indexes (Sequence): current batch indexes list, e.g. [n, n + 1, ..., n + batch_size]
            batch_size (int):
            data_size (Tuple): input image size of shape (C, H, W)

        Return:
            batch_x (Numpy ndarray): batch array of input images (B, C, H, W)
            batch_y (Numpy ndarray): batch array of ground truth lables (B,)
        """
        batch_x = np.zeros((batch_size,) + data_size)
        batch_y = np.zeros((batch_size,) + (1,), dtype=np.int32)
        for idx, i in enumerate(indexes):
            sample_x, sample_y = self.__getitem__(i)
            batch_x[idx, :, :, :] = sample_x
            batch_y[idx, :] = sample_y

        return batch_x, batch_y


class CNNModel(model.Model):

    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.input_size = 28
        self.dimension = 4
        self.num_classes = num_classes

        self.layer1 = layer.Conv2d(16, kernel_size=3, activation="RELU")
        self.bn1 = layer.BatchNorm2d()
        self.layer2 = layer.Conv2d(16, kernel_size=3, activation="RELU")
        self.bn2 = layer.BatchNorm2d()
        self.pooling2 = layer.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = layer.Conv2d(64, kernel_size=3, activation="RELU")
        self.bn3 = layer.BatchNorm2d()
        self.layer4 = layer.Conv2d(64, kernel_size=3, activation="RELU")
        self.bn4 = layer.BatchNorm2d()
        self.layer5 = layer.Conv2d(64,
                                   kernel_size=3,
                                   padding=1,
                                   activation="RELU")
        self.bn5 = layer.BatchNorm2d()
        self.pooling5 = layer.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = layer.Flatten()

        self.linear1 = layer.Linear(128)
        self.linear2 = layer.Linear(128)
        self.linear3 = layer.Linear(self.num_classes)

        self.relu = layer.ReLU()

        self.softmax_cross_entropy = layer.SoftMaxCrossEntropy()
        self.dropout = layer.Dropout(ratio=0.3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.bn1(x)
        x = self.layer2(x)
        x = self.bn2(x)
        x = self.pooling2(x)

        x = self.layer3(x)
        x = self.bn3(x)
        x = self.layer4(x)
        x = self.bn4(x)
        x = self.layer5(x)
        x = self.bn5(x)
        x = self.pooling5(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        return x

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def train_one_batch(self, x, y, dist_option, spars):
        out = self.forward(x)
        loss = self.softmax_cross_entropy(out, y)

        if dist_option == 'plain':
            self.optimizer(loss)
        elif dist_option == 'half':
            self.optimizer.backward_and_update_half(loss)
        elif dist_option == 'partialUpdate':
            self.optimizer.backward_and_partial_update(loss)
        elif dist_option == 'sparseTopK':
            self.optimizer.backward_and_sparse_update(loss,
                                                      topK=True,
                                                      spars=spars)
        elif dist_option == 'sparseThreshold':
            self.optimizer.backward_and_sparse_update(loss,
                                                      topK=False,
                                                      spars=spars)
        return out, loss


def accuracy(pred, target):
    """Compute recall accuracy.

    Args:
        pred (Numpy ndarray): Prediction array, should be in shape (B, C)
        target (Numpy ndarray): Ground truth array, should be in shape (B, )

    Return:
        correct (Float): Recall accuracy
    """
    # y is network output to be compared with ground truth (int)
    y = np.argmax(pred, axis=1)
    a = (y[:, None] == target).sum()
    correct = np.array(a, "int").sum()
    return correct


# Define pre-processing methods (transforms)
transforms = Compose(
    [ToTensor(),
     Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Dataset loading
dataset_path = "./bloodmnist"
train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "val")
cfg_path = os.path.join(dataset_path, "param.json")

with open(cfg_path, 'r') as load_f:
    num_class = json.load(load_f)["num_classes"]

train_dataset = ClassDataset(train_path, transforms)
val_dataset = ClassDataset(val_path, transforms)

batch_size = 256

# Model configuration for CNN
model = CNNModel(num_classes=num_class)
criterion = layer.SoftMaxCrossEntropy()
optimizer_ft = opt.Adam(lr=1e-3)

# Start training
dev = device.create_cpu_device()
dev.SetRandSeed(0)
np.random.seed(0)

tx = tensor.Tensor((batch_size, 3, model.input_size, model.input_size), dev,
                   singa_dtype['float32'])
ty = tensor.Tensor((batch_size,), dev, tensor.int32)

num_train_batch = train_dataset.__len__() // batch_size
num_val_batch = val_dataset.__len__() // batch_size
idx = np.arange(train_dataset.__len__(), dtype=np.int32)

model.set_optimizer(optimizer_ft)
model.compile([tx], is_train=True, use_graph=False, sequential=False)
dev.SetVerbosity(0)

max_epoch = 100
for epoch in range(max_epoch):
    print(f'Epoch {epoch}:')

    start_time = time.time()

    train_correct = np.zeros(shape=[1], dtype=np.float32)
    test_correct = np.zeros(shape=[1], dtype=np.float32)
    train_loss = np.zeros(shape=[1], dtype=np.float32)

    # Training part
    model.train()
    for b in tqdm(range(num_train_batch)):
        # Extract batch from image list
        x, y = train_dataset.batchgenerator(
            idx[b * batch_size:(b + 1) * batch_size],
            batch_size=batch_size,
            data_size=(3, model.input_size, model.input_size))
        x = x.astype(np_dtype['float32'])

        tx.copy_from_numpy(x)
        ty.copy_from_numpy(y)

        out, loss = model(tx, ty, dist_option="plain", spars=None)
        train_correct += accuracy(tensor.to_numpy(out), y)
        train_loss += tensor.to_numpy(loss)[0]
    print('Training loss = %f, training accuracy = %f' %
          (train_loss, train_correct / (num_train_batch * batch_size)))

    # Validation part
    model.eval()
    for b in tqdm(range(num_val_batch)):
        x, y = train_dataset.batchgenerator(
            idx[b * batch_size:(b + 1) * batch_size],
            batch_size=batch_size,
            data_size=(3, model.input_size, model.input_size))
        x = x.astype(np_dtype['float32'])

        tx.copy_from_numpy(x)
        ty.copy_from_numpy(y)

        out = model(tx)
        test_correct += accuracy(tensor.to_numpy(out), y)

    print('Evaluation accuracy = %f, Elapsed Time = %fs' %
          (test_correct /
           (num_val_batch * batch_size), time.time() - start_time))
