
#include <cblas.h>
#include <glog/logging.h>
#include <string>

#include "singa.h"

#include "utils/tinydir.h"

namespace singa {

void Driver::Init(int argc, char **argv) {
  google::InitGoogleLogging(argv[0]);

  //  unique job ID generated from singa-run.sh, passed in as "-singa_job <id>"
  int arg_pos = ArgPos(argc, argv, "-singa_job");
  job_id_ = (arg_pos != -1) ? atoi(argv[arg_pos+1]) : -1;

  //  global signa conf passed by singa-run.sh as "-singa_conf <path>"
  arg_pos = ArgPos(argc, argv, "-singa_conf");
  if (arg_pos != -1)
    ReadProtoFromTextFile(argv[arg_pos+1], &singa_conf_);
  else
    ReadProtoFromTextFile("conf/singa.conf", &singa_conf_);

  //  job conf passed by users as "-conf <path>"
  arg_pos = ArgPos(argc, argv, "-conf");
  CHECK_NE(arg_pos, -1);
  ReadProtoFromTextFile(argv[arg_pos+1], &job_conf_);

  // register layers
  RegisterLayer<BridgeDstLayer, int>(kBridgeDst);
  RegisterLayer<BridgeSrcLayer, int>(kBridgeSrc);
  RegisterLayer<ConvolutionLayer, int>(kConvolution);
  RegisterLayer<ConcateLayer, int>(kConcate);
  RegisterLayer<DropoutLayer, int>(kDropout);
  RegisterLayer<EuclideanLossLayer, int>(kEuclideanLoss);
  RegisterLayer<InnerProductLayer, int>(kInnerProduct);
  RegisterLayer<LabelLayer, int>(kLabel);
  RegisterLayer<LRNLayer, int>(kLRN);
  RegisterLayer<MnistLayer, int>(kMnist);
  RegisterLayer<PrefetchLayer, int>(kPrefetch);
  RegisterLayer<PoolingLayer, int>(kPooling);
  RegisterLayer<RBMHidLayer, int>(kRBMHid);
  RegisterLayer<RBMVisLayer, int>(kRBMVis);
  RegisterLayer<RGBImageLayer, int>(kRGBImage);
  RegisterLayer<ReLULayer, int>(kReLU);
  RegisterLayer<ShardDataLayer, int>(kShardData);
  RegisterLayer<SigmoidLayer, int>(kSigmoid);
  RegisterLayer<SliceLayer, int>(kSlice);
  RegisterLayer<SoftmaxLossLayer, int>(kSoftmaxLoss);
  RegisterLayer<SplitLayer, int>(kSplit);
  RegisterLayer<STanhLayer, int>(kSTanh);
#ifdef USE_LMDB
  RegisterLayer<LMDBDataLayer, int>(kLMDBData);
#endif

  // register updaters
  RegisterUpdater<AdaGradUpdater>(kAdaGrad);
  RegisterUpdater<NesterovUpdater>(kNesterov);
  // TODO(wangwei) RegisterUpdater<kRMSPropUpdater>(kRMSProp);
  RegisterUpdater<SGDUpdater>(kSGD);

  // register learning rate change methods
  RegisterLRGenerator<LRGenerator>(kFixed);
  RegisterLRGenerator<FixedStepLRGen>(kFixedStep);
  RegisterLRGenerator<StepLRGen>(kStep);
  RegisterLRGenerator<LinearLRGen>(kLinear);
  RegisterLRGenerator<ExpLRGen>(kExponential);
  RegisterLRGenerator<InvLRGen>(kInverse);
  RegisterLRGenerator<InvTLRGen>(kInverseT);

  // register workers
  RegisterWorker<BPWorker>(kBP);
  RegisterWorker<CDWorker>(kCD);

  // register params
  RegisterParam<Param>(0);

  // register param init methods
  RegisterParamGenerator<ParamGenerator>(kConstant);
  RegisterParamGenerator<GaussianGen>(kGaussian);
  RegisterParamGenerator<UniformGen>(kUniform);
  RegisterParamGenerator<GaussianSqrtFanInGen>(kGaussianSqrtFanIn);
  RegisterParamGenerator<UniformSqrtFanInGen>(kUniformSqrtFanIn);
  RegisterParamGenerator<UniformSqrtFanInOutGen>(kUniformSqrtFanInOut);
}



void Driver::Submit(bool resume, const JobProto& jobConf) {
  if (singa_conf_.has_log_dir())
    SetupLog(singa_conf_.log_dir(), std::to_string(job_id_)
             + "-" + jobConf.name());
  tinydir_dir workspace;
  if (tinydir_open(&workspace, jobConf.cluster().workspace().c_str()) == -1)
    LOG(FATAL) << "workspace does not exist: " << jobConf.cluster().workspace();
  if (jobConf.num_openblas_threads() != 1)
    LOG(WARNING) << "openblas with "
      << jobConf.num_openblas_threads() << " threads";
  openblas_set_num_threads(jobConf.num_openblas_threads());

  JobProto job;
  job.CopyFrom(jobConf);
  job.set_id(job_id_);
  Trainer trainer;
  trainer.Start(resume, singa_conf_, &job);
}

}  // namespace singa
