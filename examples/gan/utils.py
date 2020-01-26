#!/usr/bin/env python
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

import gzip
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import sys
import time

try:
    import urllib.request as ul_request
except ImportError:
    import urllib as ul_request


def print_log(s):
    t = time.ctime()
    print('[{}]{}'.format(t, s))


def load_data(filepath):
    with gzip.open(filepath, 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f, encoding='bytes')
        traindata = train_set[0].astype(np.float32)
        validdata = valid_set[0].astype(np.float32)
        testdata = test_set[0].astype(np.float32)
        trainlabel = train_set[1].astype(np.float32)
        validlabel = valid_set[1].astype(np.float32)
        testlabel = test_set[1].astype(np.float32)
        return traindata, trainlabel, validdata, validlabel, testdata, testlabel


def download_data(gzfile, url):
    if os.path.exists(gzfile):
        print('Downloaded already!')
        sys.exit(0)
    print('Downloading data %s' % (url))
    ul_request.urlretrieve(url, gzfile)
    print('Finished!')


def show_images(filepath):
    with open(filepath, 'rb') as f:
        imgs = pickle.load(f)
        r, c = 5, 5
        fig, axs = plt.subplots(5, 5)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        plt.show()
