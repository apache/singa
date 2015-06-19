#include <gflags/gflags.h>
#include <glog/logging.h>
#include "trainer/trainer.h"
#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif  // GFLAGS_GFLAGS_H_
/**
 * \file main.cc is the main entry of SINGA, like the driver program for Hadoop.
 *
 * 1. Users register their own implemented classes, e.g., layer, updater, etc.
 * 2. Users prepare the google protobuf object for the model configuration and
 * the cluster configuration.
 * 3. Users call trainer to start the training.
 *
 * TODO
 * 1. Add the resume function to continue training from a previously stopped
 * point.
 * 2. Add helper functions for users to configure their model and cluster
 * easily, e.g., AddLayer(layer_type, source_layers, meta_data).
 */

DEFINE_int32(procsID, -1, "Global process ID");
DEFINE_string(cluster, "examples/mnist/cluster.conf", "Cluster config file");
DEFINE_string(model, "examples/mnist/conv.conf", "Model config file");

/**
 * Register layers, and other customizable classes.
 *
 * If users want to use their own implemented classes, they should register
 * them here. Refer to the Worker::RegisterDefaultClasses()
 */
void RegisterClasses(const singa::ModelProto& proto){
}

int main(int argc, char **argv) {
  // TODO set log dir
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  singa::ClusterProto cluster;
  singa::ReadProtoFromTextFile(FLAGS_cluster.c_str(), &cluster);
  singa::ModelProto model;
  singa::ReadProtoFromTextFile(FLAGS_model.c_str(), &model);
  LOG(INFO)<<"The cluster config is\n"<<cluster.DebugString();
  LOG(INFO)<<"The model config is\n"<<model.DebugString();

  RegisterClasses(model);
  singa::Trainer trainer;
  trainer.Start(model, cluster, FLAGS_procsID);
  return 0;
}
