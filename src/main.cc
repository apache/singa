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
 * Users must pass at least one argument to the singa-run.sh, i.e., the job
 * configuration file which includes the cluster topology setting. Other fields
 * e.g, neuralnet, updater can be configured in main.cc.
 *
 * TODO
 * Add helper functions for users to generate their configurations easily.
 * e.g., AddLayer(layer_type, source_layers, meta_data),
 * or, MLP(layer1_size, layer2_size, tanh, loss);
 */

int main(int argc, char **argv) {
  //  must create driver at the beginning and call its Init method.
  singa::Driver driver;
  driver.Init(argc, argv);

  //  if -resume in argument list, set resume to true; otherwise false
  int resume_pos = singa::ArgPos(argc, argv, "-resume");
  bool resume = (resume_pos != -1);

  //  users can register new subclasses of layer, updater, etc.

  //  get the job conf, and custmize it if need
  singa::JobProto jobConf = driver.job_conf();

  //  submit the job
  driver.Submit(resume, jobConf);
  return 0;
}
