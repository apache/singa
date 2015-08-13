#include <gflags/gflags.h>
#include <glog/logging.h>
#include <iostream>
#include <fstream>
#include "utils/cluster_rt.h"
#include "proto/singa.pb.h"
#include "utils/common.h"
#ifndef GFLAGS_GFLAGS_H_
namespace gflags = google;
#endif  // GFLAGS_GFLAGS_H_

DEFINE_string(global, "conf/singa.conf", "Global config file");

singa::SingaProto global;
const int SUCCESS = 0;
const int ARG_ERR = 1;
const int RUN_ERR = 2;

// generate a unique job id
int create() {
  singa::JobManager mngr(global.zookeeper_host());
  if (!mngr.Init()) return RUN_ERR;
  int id;
  if (!mngr.GenerateJobID(&id)) return RUN_ERR;
  printf("%d\n", id);
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

// clean a job path in zookeeper
int clean(int id) {
  singa::JobManager mngr(global.zookeeper_host());
  if (!mngr.Init()) return RUN_ERR;
  if (!mngr.Clean(id)) return RUN_ERR;
  return SUCCESS;
}

// clean all singa data in zookeeper
int cleanup() {
  singa::JobManager mngr(global.zookeeper_host());
  if (!mngr.Init()) return RUN_ERR;
  if (!mngr.Cleanup()) return RUN_ERR;
  return SUCCESS;
}

// show log dir in global config
int getlogdir() {
  std::string dir = global.log_dir();
  while (dir.length() > 1 && dir[dir.length()-1] == '/') dir.pop_back();
  printf("%s\n", dir.c_str());
  return SUCCESS;
}

int main(int argc, char **argv) {
  std::string usage = "usage: singatool <command> <args>\n"
      " getlogdir    :  show log dir in global config\n"
      " create       :  generate a unique job id\n"
      " list         :  list running singa jobs\n"
      " listall      :  list all singa jobs\n"
      " view JOB_ID  :  view procs of a singa job\n"
      " clean JOB_ID :  clean a job path in zookeeper\n"
      " cleanup      :  clean all singa data in zookeeper\n";
  // set logging level to ERROR and log to STDERR
  FLAGS_logtostderr = 1;
  FLAGS_minloglevel = 2;
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  singa::ReadProtoFromTextFile(FLAGS_global.c_str(), &global);

  // stat code: ARG_ERR for wrong argument, RUN_ERR for runtime error
  int stat = SUCCESS;
  if (argc <= 1) stat = ARG_ERR;
  else {
    if (!strcmp(argv[1], "create"))
      stat = create();
    else if (!strcmp(argv[1], "list"))
      stat = list(false);
    else if (!strcmp(argv[1], "listall"))
      stat = list(true);
    else if (!strcmp(argv[1], "view"))
      stat = (argc > 2) ? view(atoi(argv[2])) : ARG_ERR;
    else if (!strcmp(argv[1], "clean"))
      stat = (argc > 2) ? clean(atoi(argv[2])) : ARG_ERR;
    else if (!strcmp(argv[1], "cleanup"))
      stat = cleanup();
    else if (!strcmp(argv[1], "getlogdir"))
      stat = getlogdir();
    else stat = ARG_ERR;
  }
  
  if (stat == ARG_ERR) LOG(ERROR) << usage;
  return stat;
}
