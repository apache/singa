#!/usr/bin/env python
import urllib
import tarfile
import os
import sys
import argparse


def extract_tarfile(filepath):
    if os.path.exists(filepath):
        print 'The tar file does exist. Extracting it now..'
        with tarfile.open(filepath, 'r') as f:
            f.extractall('.')
        print 'Finished!'
        sys.exit(0)


def check_dir_exist(dirpath):
    if os.path.exists(dirpath):
        print 'Directory %s does exist. To redownload the files, '\
            'remove the existing directory and %s.tar.gz' % (dirpath, dirpath)
        return True
    else:
        return False


def do_download(dirpath, gzfile, url):
    if check_dir_exist(dirpath):
        sys.exit(0)
    print 'Downloading CIFAR10 from %s' % (url)
    urllib.urlretrieve(url, gzfile)
    extract_tarfile(gzfile)
    print 'Finished!'


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
