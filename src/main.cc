#include "singa.h"
/**
 * \file main.cc provides an example main func.
 *
 * Like the main func of Hadoop, it prepares the job configuration and submit it
 * to the Driver which starts the training.
 *
 * Users can define their own main func to prepare the job configuration in
 * different ways other than reading it from a configuration file. But the main
 * func must call Driver::Init at the beginning, and pass the job configuration
 * and resume option to the Driver for job submission.
 *
 * Optionally, users can register their own implemented classes, e.g., layer,
 * updater, through the registration func provided by the Driver.
 *
 *
 * TODO
 * Add helper functions for users to generate their configurations easily.
 * e.g., AddLayer(layer_type, source_layers, meta_data),
 * or, MLP(layer1_size, layer2_size, tanh, loss);
 */

DEFINE_bool(resume, false, "Resume from checkpoint passed at cmd line");
DEFINE_string(conf, "./job.conf", "job conf passed at cmd line");

int main(int argc, char **argv) {
  //  must create driver at the beginning and call its Init method.
  singa::Driver driver;
  driver.Init(argc, argv);

  //  users can register new subclasses of layer, updater, etc.

  //  prepare job conf;
  singa::JobProto jobConf;
  singa::ReadProtoFromTextFile(FLAGS_conf.c_str(), &jobConf);

  //  submit the job
  driver.Submit(FLAGS_resume, jobConf);
  return 0;
}
