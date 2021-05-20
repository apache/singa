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

from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
import urllib.request, urllib.parse, urllib.error
import tarfile
import os
import sys


def extract_tarfile(filepath):
    if os.path.exists(filepath):
        print('The tar file does exist. Extracting it now..')
        with tarfile.open(filepath, 'r') as f:
            f.extractall('/tmp/')
        print('Finished!')
        sys.exit(0)


def do_download(dirpath, gzfile, url):
    print('Downloading CIFAR from %s' % (url))
    urllib.request.urlretrieve(url, gzfile)
    extract_tarfile(gzfile)
    print('Finished!')


if __name__ == '__main__':
    dirpath = '/tmp/'
    gzfile = dirpath + 'cifar-10-python.tar.gz'
    url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
    do_download(dirpath, gzfile, url)
