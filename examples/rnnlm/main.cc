#include <string>
#include "singa.h"
#include "rnnlm.h"
#include "rnnlm.pb.h"

int main(int argc, char **argv) {
  singa::Driver driver;
  driver.Init(argc, argv);

  //if -resume in argument list, set resume to true; otherwise false
  int resume_pos = singa::ArgPos(argc, argv, "-resume");
  bool resume = (resume_pos != -1);

  //  register all layers for rnnlm
  driver.RegisterLayer<singa::EmbeddingLayer, std::string>("kEmbedding");
  driver.RegisterLayer<singa::HiddenLayer, std::string>("kHidden");
  driver.RegisterLayer<singa::OutputLayer, std::string>("kOutput");

  singa::JobProto jobConf = driver.job_conf();

  driver.Submit(resume, jobConf);
  return 0;
}
