#include <gflags/gflags.h>
#include <glog/logging.h>
#include "proto/cluster.pb.h"
#include "utils/cluster_rt.h"
#include "utils/common.h"
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif  // GFLAGS_GFLAGS_H_

DEFINE_string(global, "conf/singa.conf", "Global config file");

int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  singa::GlobalProto global;
  singa::ReadProtoFromTextFile(FLAGS_global.c_str(), &global);
  singa::SetupLog(global.log_dir(), "SingaTool");

  LOG(INFO) << "The global config is \n" << global.DebugString();

  singa::JobManager mng(global.zookeeper_host());
  int ret = 0;
  if (!mng.Init()) ret = 1;
  if (!mng.Clean()) ret = 1;
  if (ret) LOG(ERROR) << "errors in SingaTool!";
  return ret;
}
