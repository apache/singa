#include "singa.h"
/**
 * \file main.cc is the main entry of SINGA, like the driver program for Hadoop.
 *
 * 1. Users register their own implemented classes, e.g., layer, updater, etc.
 * 2. Users prepare the google protobuf object for the job configuration.
 * 3. Users call trainer to start the training.
 *
 * TODO
 * 1. Add helper functions for users to configure their model easily,
 * e.g., AddLayer(layer_type, source_layers, meta_data).
 */

DEFINE_int32(job, -1, "Unique job ID generated from singa-run.sh");
DEFINE_bool(resume, false, "Resume from checkpoint passed at cmd line");
DEFINE_string(workspace, "./workspace", "workspace passed at cmd line");

/**
 * Register layers, and other customizable classes.
 *
 * If users want to use their own implemented classes, they should register
 * them here. Refer to the Worker::RegisterDefaultClasses()
 */
void RegisterClasses() {

}


int main(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  singa::JobProto jobConf;
  std::string job_file = FLAGS_workspace + "/job.conf";
  singa::ReadProtoFromTextFile(job_file.c_str(), &jobConf);
  CHECK(jobConf.has_cluster());
  CHECK(jobConf.has_model());
  if (!jobConf.cluster().has_workspace())
    jobConf.mutable_cluster()->set_workspace(FLAGS_workspace);

  RegisterClasses();
  singa::SubmitJob(FLAGS_job, FLAGS_resume, jobConf);
  return 0;
}
