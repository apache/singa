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
from singa import singa_wrap as singa
from singa import device
from singa import tensor
from singa import opt
from singa import layer
from singa.layer import Layer
from singa import model

import numpy as np
import time
import argparse
from PIL import Image
import os
from glob import glob
from sklearn.metrics import accuracy_score
import json
from tqdm import tqdm

from transforms import Compose, ToTensor, Normalize

np_dtype = {"float16": np.float16, "float32": np.float32}

singa_dtype = {"float16": tensor.float16, "float32": tensor.float32}

# %%
transforms = Compose([
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# %%
class ClassDataset:
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
        batch_x = np.zeros((batch_size,) + data_size)
        batch_y = np.zeros((batch_size,) + (1,), dtype=np.int32)
        for idx, i in enumerate(indexes):
            sample_x, sample_y = self.__getitem__(i)
            batch_x[idx, :, :, :] = sample_x
            batch_y[idx, :] = sample_y

        return batch_x, batch_y



# %%
class Sequential:
    def __init__(self, *args):
        # super(Sequential, self).__init__()
        self.mod_list = []
        for i in args:
            self.mod_list.append(i)
    def __call__(self, input):
        for module in self.mod_list:
            input = module(input)
        return input

class CNNModel(model.Model):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        self.input_size = 28
        self.dimension = 4
        self.num_classes = num_classes
        
        # self.layer1 = Sequential(
        #     layer.Conv2d(16, kernel_size=3),
        #     layer.BatchNorm2d(),
        #     layer.ReLU())
        self.layer1 = layer.Conv2d(16, kernel_size=3, activation="RELU")
        self.bn1 = layer.BatchNorm2d()
        # self.layer2 = Sequential(
        #     layer.Conv2d(16, kernel_size=3),
        #     layer.BatchNorm2d(),
        #     layer.ReLU(),
        #     layer.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = layer.Conv2d(16, kernel_size=3, activation="RELU")
        self.bn2 = layer.BatchNorm2d()        
        self.pooling2 = layer.MaxPool2d(kernel_size=2, stride=2)

        # self.layer3 = Sequential(
        #     layer.Conv2d(64, kernel_size=3),
        #     layer.BatchNorm2d(),
        #     layer.ReLU())
        self.layer3 = layer.Conv2d(64, kernel_size=3, activation="RELU")
        self.bn3 = layer.BatchNorm2d()
        
        # self.layer4 = Sequential(
        #     layer.Conv2d(64, kernel_size=3),
        #     layer.BatchNorm2d(),
        #     layer.ReLU())
        self.layer4 = layer.Conv2d(64, kernel_size=3, activation="RELU")
        self.bn4 = layer.BatchNorm2d()

        # self.layer5 = Sequential(
        #     layer.Conv2d(64, kernel_size=3, padding=1),
        #     layer.BatchNorm2d(),
        #     layer.ReLU(),
        #     layer.MaxPool2d(kernel_size=2, stride=2))
        self.layer5 = layer.Conv2d(64, kernel_size=3, padding=1, activation="RELU")
        self.bn5 = layer.BatchNorm2d()
        self.pooling5 = layer.MaxPool2d(kernel_size=2, stride=2)

        self.flatten = layer.Flatten()

        # self.fc = Sequential(
        #     layer.Linear(128),
        #     layer.ReLU(),
        #     layer.Linear(128),
        #     layer.ReLU(),
        #     layer.Linear(num_classes))
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
        # x = self.fc(x)
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

# %%
def getACC(y_true, y_score):
    '''Accuracy metric.
    :param y_true: the ground truth labels, shape: (n_samples, n_labels) or (n_samples,) if n_labels==1
    :param y_score: the predicted score of each class,
    shape: (n_samples, n_labels) or (n_samples, n_classes) or (n_samples,) if n_labels==1 or n_classes==1
    :param task: the task of current dataset
    :param threshold: the threshold for multilabel and binary-class tasks
    '''
    y_true = y_true.squeeze()
    y_score = y_score.squeeze()

    
    ret = accuracy_score(y_true, np.argmax(y_score, axis=-1))

    return ret
def accuracy(pred, target):
    # y is network output to be compared with ground truth (int)
    y = np.argmax(pred, axis=1)
    a = (y[:,None]==target).sum()
    correct = np.array(a, "int").sum()
    return correct

# %% dataset configuration
dataset_path = "./bloodmnist"
train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "val") 
cfg_path = os.path.join(dataset_path, "param.json")

with open(cfg_path,'r') as load_f:
    num_class = json.load(load_f)["num_classes"]

train_dataset = ClassDataset(train_path, transforms)
val_dataset = ClassDataset(val_path, transforms)

batch_size = 256

# %% model configuration
model = CNNModel(num_classes=num_class)
criterion = layer.SoftMaxCrossEntropy()
# optimizer_ft = opt.SGD(lr=0.005, momentum=0.9, weight_decay=1e-5, dtype=singa_dtype["float32"])
optimizer_ft = opt.Adam(lr=1e-3)
# optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
# lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='max', patience=5, threshold=1e-3)

# %% start training
dev = device.create_cpu_device()
dev.SetRandSeed(0)
np.random.seed(0)

tx = tensor.Tensor(
        (batch_size, 3, model.input_size, model.input_size), dev,
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
    print('start**************************************************')
    
    start_time = time.time()
    # np.random.shuffle(idx)

    train_correct = np.zeros(shape=[1], dtype=np.float32)
    test_correct = np.zeros(shape=[1], dtype=np.float32)
    train_loss = np.zeros(shape=[1], dtype=np.float32)

    model.train()
    for b in tqdm(range(num_train_batch)):
        # forward + backward + optimize

        x, y = train_dataset.batchgenerator(idx[b * batch_size:(b + 1) * batch_size], 
            batch_size=batch_size, data_size=(3, model.input_size, model.input_size))
        x = x.astype(np_dtype['float32'])
        # print(tx.size())
        # print(y.dtype)

        tx.copy_from_numpy(x)
        ty.copy_from_numpy(y)

        out, loss = model(tx, ty, dist_option="plain", spars=None)
        # print(out)
        # print(loss)
        # train_correct += getACC(y, tensor.to_numpy(out))
        train_correct += accuracy(tensor.to_numpy(out), y)
        train_loss += tensor.to_numpy(loss)[0]
    print('Training loss = %f, training accuracy = %f' %
                  (train_loss, train_correct /
                   (num_train_batch * batch_size)))
    model.eval()

    for b in tqdm(range(num_val_batch)):
        # forward + backward + optimize
        x, y = train_dataset.batchgenerator(idx[b * batch_size:(b + 1) * batch_size], 
            batch_size=batch_size, data_size=(3, model.input_size, model.input_size))
        x = x.astype(np_dtype['float32'])

        tx.copy_from_numpy(x)
        ty.copy_from_numpy(y)

        out = model(tx)

        # test_correct += getACC(y, tensor.to_numpy(out))
        test_correct += accuracy(tensor.to_numpy(out), y)
    
    print('Evaluation accuracy = %f, Elapsed Time = %fs' %
                  (test_correct / (num_val_batch * batch_size),
                   time.time() - start_time))
