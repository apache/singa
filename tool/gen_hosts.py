#!/usr/bin/env python

import argparse
import os
import sys
from google.protobuf import text_format
from pb2.job_pb2 import JobProto

# parse command line
parser = argparse.ArgumentParser(description='Generate host list from host file for a SINGA job')
parser.add_argument('-conf', dest='conf', metavar='CONF_FILE', required=True, help='job.conf file')
parser.add_argument('-hosts', dest='hosts', metavar='HOST_FILE', required=True, help='global host file')
parser.add_argument('-output', dest='output', metavar='OUTPUT_FILE', required=True, help='generated list')
args = parser.parse_args();

# read from .conf file
fd_conf = open(args.conf, 'r')
job = JobProto()
text_format.Merge(str(fd_conf.read()), job)
cluster = job.cluster
nworker_procs = cluster.nworker_groups * cluster.nworkers_per_group / cluster.nworkers_per_procs
nserver_procs = cluster.nserver_groups * cluster.nservers_per_group / cluster.nservers_per_procs
nprocs = 0
if (cluster.server_worker_separate) :
  nprocs = nworker_procs+nserver_procs
else:
  nprocs = max(nworker_procs, nserver_procs)
fd_conf.close()

# read from source host file
fd_hosts = open(args.hosts, 'r')
hosts = []
for line in fd_hosts:
  line = line.strip()
  if len(line) == 0 or line[0] == '#':
    continue
  hosts.append(line)
fd_hosts.close()

# write to output file
num_hosts = len(hosts)
if (num_hosts == 0):
  print "Contains no valid host %s" % args.hosts
  sys.exit(1)
fd_output = open(args.output, 'w')
for i in range(nprocs):
  fd_output.write(hosts[i % num_hosts] + '\n')
fd_output.close()
print 'Generate host list to %s' % args.output
