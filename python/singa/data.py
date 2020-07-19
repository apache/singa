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
'''
This module includes classes for loading and prefetching data batches.

Example usage::

    import image_tool
    from PIL import Image
    from singa.data import ImageBatchIter

    tool = image_tool.ImageTool()

    def image_transform(img_path):
        global tool
        return tool.load(img_path).resize_by_range(
            (112, 128)).random_crop(
            (96, 96)).flip().get()

    data = ImageBatchIter('train.txt', 3,
                          image_transform, shuffle=True, delimiter=',',
                          image_folder='images/',
                          capacity=10)
    data.start()
    # imgs is a numpy array for a batch of images,
    # shape: batch_size, 3 (RGB), height, width
    imgs, labels = data.next()

    # convert numpy array back into images
    for idx in range(imgs.shape[0]):
        img = Image.fromarray(imgs[idx].astype(np.uint8).transpose(1, 2, 0),
                              'RGB')
        img.save('img%d.png' % idx)
    data.end()
'''
from __future__ import print_function
from __future__ import absolute_import

from builtins import range
from builtins import object
import os
import random
import time
from multiprocessing import Process, Queue
import numpy as np


class ImageBatchIter(object):
    '''Utility for iterating over an image dataset to get mini-batches.

    Args:
        img_list_file(str): name of the file containing image meta data; each
                            line consists of image_path_suffix delimiter meta_info,
                            where meta info could be label index or label strings, etc.
                            meta_info should not contain the delimiter. If the meta_info
                            of each image is just the label index, then we will parse the
                            label index into a numpy array with length=batchsize
                            (for compatibility); otherwise, we return a list of meta_info;
                            if meta info is available, we return a list of None.
        batch_size(int): num of samples in one mini-batch
        image_transform: a function for image augmentation; it accepts the full
                        image path and outputs a list of augmented images.
        shuffle(boolean): True for shuffling images in the list
        delimiter(char): delimiter between image_path_suffix and label, e.g.,
                         space or comma
        image_folder(boolean): prefix of the image path
        capacity(int): the max num of mini-batches in the internal queue.
    '''

    def __init__(self,
                 img_list_file,
                 batch_size,
                 image_transform,
                 shuffle=True,
                 delimiter=' ',
                 image_folder=None,
                 capacity=10):
        self.img_list_file = img_list_file
        self.queue = Queue(capacity)
        self.batch_size = batch_size
        self.image_transform = image_transform
        self.shuffle = shuffle
        self.delimiter = delimiter
        self.image_folder = image_folder
        self.stop = False
        self.p = None
        with open(img_list_file, 'r') as fd:
            self.num_samples = len(fd.readlines())

    def start(self):
        self.p = Process(target=self.run)
        self.p.start()
        return

    def __next__(self):
        assert self.p is not None, 'call start before next'
        while self.queue.empty():
            time.sleep(0.1)
        x, y = self.queue.get()  # dequeue one mini-batch
        return x, y

    def end(self):
        if self.p is not None:
            self.stop = True
            time.sleep(0.1)
            self.p.terminate()

    def run(self):
        img_list = []
        is_label_index = True
        for line in open(self.img_list_file, 'r'):
            item = line.strip('\n').split(self.delimiter)
            if len(item) < 2:
                is_label_index = False
                img_list.append((item[0].strip(), None))
            else:
                if not item[1].strip().isdigit():
                    # the meta info is not label index
                    is_label_index = False
                img_list.append((item[0].strip(), item[1].strip()))
        index = 0  # index for the image
        if self.shuffle:
            random.shuffle(img_list)
        while not self.stop:
            if not self.queue.full():
                x, y = [], []
                i = 0
                while i < self.batch_size:
                    img_path, img_meta = img_list[index]
                    aug_images = self.image_transform(
                        os.path.join(self.image_folder, img_path))
                    assert i + len(aug_images) <= self.batch_size, \
                        'too many images (%d) in a batch (%d)' % \
                        (i + len(aug_images), self.batch_size)
                    for img in aug_images:
                        ary = np.asarray(img.convert('RGB'), dtype=np.float32)
                        x.append(ary.transpose(2, 0, 1))
                        if is_label_index:
                            y.append(int(img_meta))
                        else:
                            y.append(img_meta)
                        i += 1
                    index += 1
                    if index == self.num_samples:
                        index = 0  # reset to the first image
                        if self.shuffle:
                            random.shuffle(img_list)
                # enqueue one mini-batch
                if is_label_index:
                    self.queue.put((np.asarray(x), np.asarray(y,
                                                              dtype=np.int32)))
                else:
                    self.queue.put((np.asarray(x), y))
            else:
                time.sleep(0.1)
        return


if __name__ == '__main__':
    from . import image_tool
    from PIL import Image
    tool = image_tool.ImageTool()

    def image_transform(img_path):
        global tool
        return tool.load(img_path).resize_by_range((112, 128)).random_crop(
            (96, 96)).flip().get()

    data = ImageBatchIter('train.txt',
                          3,
                          image_transform,
                          shuffle=False,
                          delimiter=',',
                          image_folder='images/',
                          capacity=10)
    data.start()
    imgs, labels = next(data)
    print(labels)
    for idx in range(imgs.shape[0]):
        img = Image.fromarray(imgs[idx].astype(np.uint8).transpose(1, 2, 0),
                              'RGB')
        img.save('img%d.png' % idx)
    data.end()
