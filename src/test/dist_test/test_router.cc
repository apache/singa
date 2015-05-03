#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include "utils/router.h"
#include "utils/common.h"
#include "utils/cluster.h"
DEFINE_string(hostfile, "examples/imagenet12/hostfile", "hostfile");
DEFINE_string(cluster_conf, "examples/imagenet12/cluster.conf",
    "configuration file for the cluster");
DEFINE_int32(procsID, 0, "global process ID");

int main(int argc, char** argv){
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // Init Cluster
  singa::ClusterProto pcluster;
  singa::ReadProtoFromTextFile(FLAGS_cluster_conf.c_str(), &pcluster);
  auto cluster=singa::Cluster::Get(pcluster, FLAGS_hostfile, FLAGS_procsID);
  if(cluster->AmIServer()){
    singa::Router server(5732);
    CHECK(server.Bind(cluster->server_addr(0), cluster->nworkers()));
  }else{
    singa::Router worker(5732);
    CHECK(worker.Connect(cluster->server_addr(0)));
  }
  return 0;
}
