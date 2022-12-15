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
import argparse


def extract_tarfile(filepath):
    if os.path.exists(filepath):
        print('The tar file does exist. Extracting it now..')
        with tarfile.open(filepath, 'r') as f:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, ".")
        print('Finished!')
        sys.exit(0)


def check_dir_exist(dirpath):
    if os.path.exists(dirpath):
        print('Directory %s does exist. To redownload the files, '\
            'remove the existing directory and %s.tar.gz' % (dirpath, dirpath))
        return True
    else:
        return False


def do_download(dirpath, gzfile, url):
    if check_dir_exist(dirpath):
        sys.exit(0)
    print('Downloading CIFAR10 from %s' % (url))
    urllib.request.urlretrieve(url, gzfile)
    extract_tarfile(gzfile)
    print('Finished!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Cifar10 datasets')
    parser.add_argument(
        'file',
        type=str,
        choices=['py', 'bin'])
    args = parser.parse_args()
    if args.file == 'bin':
        dirpath = 'cifar-10-batches-bin'
        gzfile = 'cifar-10-binary' + '.tar.gz'
        url = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
        do_download(dirpath, gzfile, url)
    else:
        dirpath = 'cifar-10-batches-py'
        gzfile = 'cifar-10-python' + '.tar.gz'
        url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        do_download(dirpath, gzfile, url)
