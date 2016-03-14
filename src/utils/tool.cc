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
#include <algorithm>
#include <string>
#include <vector>
#include "singa/proto/singa.pb.h"
#include "singa/utils/cluster_rt.h"
#include "singa/utils/common.h"

std::string conf_dir;
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

// generate a host list
int genhost(char* job_conf) {
  singa::JobManager mngr(global.zookeeper_host());
  if (!mngr.Init()) return RUN_ERR;
  std::vector<std::string> list;
  if (!mngr.GenerateHostList((conf_dir+"/hostfile").c_str(), job_conf, &list))
    return RUN_ERR;
  // output selected hosts
  for (std::string host : list)
    printf("%s\n", host.c_str());
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
      " cleanup            :  clean all singa data in zookeeper\n"
      "[optional arguments] NOTICE: must put at end of a command\n"
      " -confdir <dir>     :  path to singa global conf dir";

  // set logging level to ERROR and log to STDERR only
  google::LogToStderr();
  google::SetStderrLogging(google::ERROR);
  google::InitGoogleLogging(argv[0]);
  // parse -confdir argument
  int arg_pos = singa::ArgPos(argc, argv, "-confdir");
  conf_dir = arg_pos == -1 ? "conf" : argv[arg_pos+1];
  if (arg_pos != -1) argc -= 2;
  singa::ReadProtoFromTextFile((conf_dir+"/singa.conf").c_str(), &global);

  // stat code: ARG_ERR for wrong argument, RUN_ERR for runtime error
  int stat = (argc <= 1) ? ARG_ERR : SUCCESS;
  if (stat == SUCCESS) {
    if (!strcmp(argv[1], "getlogdir"))
      stat = getlogdir();
    else if (!strcmp(argv[1], "create"))
      stat = create();
    else if (!strcmp(argv[1], "genhost"))
      stat = (argc > 2) ? genhost(argv[2]) : genhost(nullptr);
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
