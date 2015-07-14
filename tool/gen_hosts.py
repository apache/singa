#!/usr/bin/env python

import argparse
import os
import sys
from google.protobuf import text_format
from plot.cluster_pb2 import ClusterProto

# parse command line
parser = argparse.ArgumentParser(description='Generate host list from host file for a SINGA job')
parser.add_argument('-conf', dest='conf', metavar='CONF_FILE', required=True, help='cluster.conf file')
parser.add_argument('-src', dest='src', metavar='SRC_FILE', required=True, help='global host file')
parser.add_argument('-dst', dest='dst', metavar='DST_FILE', required=True, help='generated list')
args = parser.parse_args();

# change to SINGA_HOME
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname+'/..')

# read from .conf file
fd_conf = open(args.conf, 'r')
cluster = ClusterProto()
text_format.Merge(str(fd_conf.read()), cluster)
nworker_procs = cluster.nworker_groups * cluster.nworkers_per_group / cluster.nworkers_per_procs
nserver_procs = cluster.nserver_groups * cluster.nservers_per_group / cluster.nservers_per_procs
nprocs = 0
if (cluster.server_worker_separate) :
  nprocs = nworker_procs+nserver_procs
else:
  nprocs = max(nworker_procs, nserver_procs)
fd_conf.close()

# read from source host file
fd_src = open(args.src, 'r')
hosts = []
for line in fd_src:
  line = line.strip()
  if len(line) == 0 or line[0] == '#':
    continue
  hosts.append(line)
fd_src.close()

# write to dst file
num_hosts = len(hosts)
if (num_hosts == 0):
  print 'source host file is empty'
  sys.exit()
fd_dst = open(args.dst, 'w')
for i in range(nprocs):
  fd_dst.write(hosts[i % num_hosts] + '\n')
fd_dst.close()
