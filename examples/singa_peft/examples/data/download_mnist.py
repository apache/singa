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

import argparse
import os
import urllib.request


def check_exist_or_download(url, download_dir):
    os.makedirs(download_dir, exist_ok=True)

    name = url.rsplit('/', 1)[-1]
    filename = os.path.join(download_dir, name)

    if not os.path.isfile(filename):
        print("Downloading %s to %s" % (url, filename))
        urllib.request.urlretrieve(url, filename)
    else:
        print("Already Downloaded: %s" % filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Download the MNIST dataset.'
    )
    parser.add_argument(
        '-dir',
        '--dir-path',
        dest='dir_path',
        default='/tmp/mnist',
        help='Directory to save the MNIST dataset.'
    )
    args = parser.parse_args()

    train_x_url = 'https://github.com/fgnt/mnist/raw/master/train-images-idx3-ubyte.gz'
    train_y_url = 'https://github.com/fgnt/mnist/raw/master/train-labels-idx1-ubyte.gz'
    valid_x_url = 'https://github.com/fgnt/mnist/raw/master/t10k-images-idx3-ubyte.gz'
    valid_y_url = 'https://github.com/fgnt/mnist/raw/master/t10k-labels-idx1-ubyte.gz'

    check_exist_or_download(train_x_url, args.dir_path)
    check_exist_or_download(train_y_url, args.dir_path)
    check_exist_or_download(valid_x_url, args.dir_path)
    check_exist_or_download(valid_y_url, args.dir_path)
