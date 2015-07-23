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

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  // set logging level to ERROR and log to STDERR
  FLAGS_logtostderr = 1;
  FLAGS_minloglevel = 2;
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  singa::SingaProto global;
  singa::ReadProtoFromTextFile(FLAGS_global.c_str(), &global);
  LOG(INFO) << "The global config is \n" << global.DebugString();

  singa::JobManager mng(global.zookeeper_host());
  std::string usage = "singatool usage:\n"
      "# ./singatool create       :  generate a unique job id\n"
      "# ./singatool list         :  list running singa jobs\n"
      "# ./singatool view JOB_ID  :  view procs of a singa job\n"
      "# ./singatool clean JOB_ID :  clean a job path in zookeeper\n"
      "# ./singatool cleanup      :  clean all singa data in zookeeper\n"
      "# ./singatool listall      :  list all singa jobs\n";
  if (argc <= 1) {
    LOG(ERROR) << usage;
    return 1;
  }
  if (!mng.Init()) return 1;
  if (!strcmp(argv[1], "create")) {
    int id = mng.GenerateJobID();
    printf("%d\n", id);
  } else if (!strcmp(argv[1], "list")) {
    std::vector<singa::JobInfo> jobs;
    if (!mng.ListJobs(&jobs)) return 1;
    printf("JOB ID    |NUM PROCS  \n");
    printf("----------|-----------\n");
    for (singa::JobInfo job : jobs) {
      if (!job.procs) continue;
      printf("job-%-6d|%-10d\n", job.id, job.procs);
    }
  } else if (!strcmp(argv[1], "listall")) {
    std::vector<singa::JobInfo> jobs;
    if (!mng.ListJobs(&jobs)) return 1;
    printf("JOB ID    |NUM PROCS  \n");
    printf("----------|-----------\n");
    for (singa::JobInfo job : jobs) {
      printf("job-%-6d|%-10d\n", job.id, job.procs);
    }
  } else if (!strcmp(argv[1], "view")) {
    if (argc <= 2) {
      LOG(ERROR) << usage;
      return 1;
    }
    int id = atoi(argv[2]);
    std::vector<std::string> procs;
    if (!mng.ListJobProcs(id, &procs)) return 1;
    for (std::string s : procs) {
      printf("%s\n", s.c_str());
    }
  } else if (!strcmp(argv[1], "clean")) {
    if (argc <= 2) {
      LOG(ERROR) << usage;
      return 1;
    }
    int id = atoi(argv[2]);
    if (!mng.Clean(id)) return 1;
  } else if (!strcmp(argv[1], "cleanup")) {
    if (!mng.Cleanup()) return 1;
  } else {
    LOG(ERROR) << usage;
    return 1;
  }

  return 0;
}
