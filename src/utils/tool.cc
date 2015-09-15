/************************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
* 
*   http://www.apache.org/licenses/LICENSE-2.0
* 
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/

#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include "proto/job.pb.h"
#include "proto/singa.pb.h"
#include "utils/cluster_rt.h"
#include "utils/common.h"

singa::SingaProto global;
const int SUCCESS = 0;
const int ARG_ERR = 1;
const int RUN_ERR = 2;

// show log dir in global config
int getlogdir() {
  std::string dir = global.log_dir();
  while (dir.length() > 1 && dir[dir.length()-1] == '/') dir.pop_back();
  printf("%s\n", dir.c_str());
  return SUCCESS;
}

// generate a unique job id
int create() {
  singa::JobManager mngr(global.zookeeper_host());
  if (!mngr.Init()) return RUN_ERR;
  int id;
  if (!mngr.GenerateJobID(&id)) return RUN_ERR;
  printf("%d\n", id);
  return SUCCESS;
}

// extract cluster configuration part from the job config file
// TODO improve this function to make it robust
const std::string extract_cluster(const char* jobfile) {
  std::ifstream fin;
  fin.open(jobfile, std::ifstream::in);
  CHECK(fin.is_open()) << "cannot open job conf file " << jobfile;
  std::string line;
  std::string cluster;
  while (std::getline(fin, line)) {
    // end of extraction (cluster config has not nested messages)
    if (line.find("}") != std::string::npos && cluster.length()) {
      cluster += line.substr(0, line.find("}"));
      break;
    }
    unsigned int pos = 0;
    while (pos < line.length() && line.at(pos) == ' ' ) pos++;
    if (line.find("cluster", pos) == pos) {  // start with <whitespace> cluster
      pos += 7;
      do {  // looking for the first '{', which may be in the next lines
        while (pos < line.length() &&
            (line.at(pos) == ' ' || line.at(pos) =='\t')) pos++;
        if (pos < line.length()) {
          CHECK_EQ(line.at(pos), '{') << "error around 'cluster' field";
          cluster =  " ";  // start extraction
          break;
        } else
          pos = 0;
      }while(std::getline(fin, line));
    } else if (cluster.length()) {
        cluster += line + "\n";
    }
  }
  return cluster;
}


// generate a host list
int genhost(char* job_conf) {
  // compute required #process from job conf
  singa::ClusterProto cluster;
  google::protobuf::TextFormat::ParseFromString(extract_cluster(job_conf),
      &cluster);
  int nworker_procs = cluster.nworker_groups() * cluster.nworkers_per_group()
                      / cluster.nworkers_per_procs();
  int nserver_procs = cluster.nserver_groups() * cluster.nservers_per_group()
                      / cluster.nservers_per_procs();
  int nprocs = 0;
  if (cluster.server_worker_separate())
    nprocs = nworker_procs + nserver_procs;
  else
    nprocs = std::max(nworker_procs, nserver_procs);

  // get available host list from global conf
  std::fstream hostfile("conf/hostfile");
  if (!hostfile.is_open()) {
    LOG(ERROR) << "Cannot open file: " << "conf/hostfile";
    return RUN_ERR;
  }
  std::vector<std::string> hosts;
  std::string host;
  while (!hostfile.eof()) {
    getline(hostfile, host);
    if (!host.length() || host[0] == '#') continue;
    hosts.push_back(host);
  }
  if (!hosts.size()) {
    LOG(ERROR) << "Empty host file";
    return RUN_ERR;
  }
  // output selected hosts
  for (int i = 0; i < nprocs; ++i)
    printf("%s\n", hosts[i % hosts.size()].c_str());
  return SUCCESS;
}

// list singa jobs (running or all)
int list(bool all) {
  singa::JobManager mngr(global.zookeeper_host());
  if (!mngr.Init()) return RUN_ERR;
  std::vector<singa::JobInfo> jobs;
  if (!mngr.ListJobs(&jobs)) return RUN_ERR;
  printf("JOB ID    |NUM PROCS  \n");
  printf("----------|-----------\n");
  for (singa::JobInfo job : jobs) {
    if (!job.procs && !all) continue;
    printf("%-10d|%-10d\n", job.id, job.procs);
  }
  return SUCCESS;
}

// view procs of a singa job
int view(int id) {
  singa::JobManager mngr(global.zookeeper_host());
  if (!mngr.Init()) return RUN_ERR;
  std::vector<std::string> procs;
  if (!mngr.ListJobProcs(id, &procs)) return RUN_ERR;
  for (std::string s : procs) {
    printf("%s\n", s.c_str());
  }
  return SUCCESS;
}

// remove a job path in zookeeper
int remove(int id) {
  singa::JobManager mngr(global.zookeeper_host());
  if (!mngr.Init()) return RUN_ERR;
  if (!mngr.Remove(id)) return RUN_ERR;
  return SUCCESS;
}

// remove all job paths in zookeeper
int removeall() {
  singa::JobManager mngr(global.zookeeper_host());
  if (!mngr.Init()) return RUN_ERR;
  if (!mngr.RemoveAllJobs()) return RUN_ERR;
  return SUCCESS;
}

// clean all singa data in zookeeper
int cleanup() {
  singa::JobManager mngr(global.zookeeper_host());
  if (!mngr.Init()) return RUN_ERR;
  if (!mngr.CleanUp()) return RUN_ERR;
  return SUCCESS;
}

int main(int argc, char **argv) {
  std::string usage = "Usage: singatool <command> <args>\n"
      " getlogdir          :  show log dir in global config\n"
      " create             :  generate a unique job id\n"
      " genhost <job conf> :  generate a host list\n"
      " list               :  list running singa jobs\n"
      " listall            :  list all singa jobs\n"
      " view <job id>      :  view procs of a singa job\n"
      " remove <job id>    :  remove a job path in zookeeper\n"
      " removeall          :  remova all job paths in zookeeper\n"
      " cleanup            :  clean all singa data in zookeeper\n";
  // set logging level to ERROR and log to STDERR only
  google::LogToStderr();
  google::SetStderrLogging(google::ERROR);
  google::InitGoogleLogging(argv[0]);
  singa::ReadProtoFromTextFile("conf/singa.conf", &global);

  // stat code: ARG_ERR for wrong argument, RUN_ERR for runtime error
  int stat = (argc <= 1) ? ARG_ERR : SUCCESS;
  if (stat == SUCCESS) {
    if (!strcmp(argv[1], "getlogdir"))
      stat = getlogdir();
    else if (!strcmp(argv[1], "create"))
      stat = create();
    else if (!strcmp(argv[1], "genhost"))
      stat = (argc > 2) ? genhost(argv[2]) : ARG_ERR;
    else if (!strcmp(argv[1], "list"))
      stat = list(false);
    else if (!strcmp(argv[1], "listall"))
      stat = list(true);
    else if (!strcmp(argv[1], "view"))
      stat = (argc > 2) ? view(atoi(argv[2])) : ARG_ERR;
    else if (!strcmp(argv[1], "remove"))
      stat = (argc > 2) ? remove(atoi(argv[2])) : ARG_ERR;
    else if (!strcmp(argv[1], "removeall"))
      stat = removeall();
    else if (!strcmp(argv[1], "cleanup"))
      stat = cleanup();
    else
      stat = ARG_ERR;
  }

  if (stat == ARG_ERR) LOG(ERROR) << usage;
  return stat;
}
