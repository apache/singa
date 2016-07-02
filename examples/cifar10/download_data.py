#!/usr/bin/env python
import urllib
import tarfile
import os
import sys


def extract_tarfile(filepath):
    with tarfile.open(filepath, 'r') as f:
        f.extractall('.')


dirpath = 'cifar-10-batches-bin'
gzfile = 'cifar-10-binary' + '.tar.gz'
if os.path.exists(dirpath):
    print 'Directory %s does exist. To redownload the files, '\
        'remove the existing directory and %s.tar.gz' % (dirpath, dirpath)
    sys.exit(0)

if os.path.exists(gzfile):
    print 'The tar file does exist. Extracting it now..'
    extract_tarfile(gzfile)
    print 'Finished!'
    sys.exit(0)

url = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
print 'Downloading CIFAR10 from %s' % (url)
urllib.urlretrieve(url, gzfile)
extract_tarfile(gzfile)
print 'Finished!'
