/************************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/
#ifndef SINGA_SINGA_DRIVER_H_
#define SINGA_SINGA_DRIVER_H_

#include <vector>
#include "singa/proto/job.pb.h"
#include "singa/proto/singa.pb.h"
#include "singa/utils/factory.h"
#include "singa/utils/param.h"
#include "singa/utils/singleton.h"
#include "singa/utils/updater.h"
#include "singa/neuralnet/layer.h"
#include "singa/worker.h"
#include "singa/server.h"

namespace singa {
using std::vector;
class Driver {
 public:
  /**
   * Init SINGA
   * - init glog
   * - parse job id and job conf from cmd line
   * - register built-in layer, worker, updater, param subclasses.
   *
   * May be used for MPI init if it is used for message passing.
   */
  void Init(int argc, char** argv);
  /**
   * Init SINGA LOG
   * Used for python binding. Users can also directly call it as a C++ API.
   * - init glog with given parameters
   *
   */   
  void InitLog(char *arg);
  /**
   * Update job configuration and call Train(const JobProto&) to start the
   * training.
   *
   * It sets up the logging path and checkpoing files (if resume), and checks
   * the existence of the workspace folder .
   *
   * @param[in] resume if true resume the training from the latest checkpoint
   * files.
   * @param[in] job_conf job configuration.
   */
  void Train(bool resume, const JobProto& job_conf);
  /**
   * Used for python binding. Users can also directly call it as a C++ API.
   *
   * It completes the functions as defined above but accept serialized string
   * parameters.
   *
   * @param[in] resume if true resume the training from the latest checkpoint
   * files.
   * @param[in] str serialized string recorded job configuration.
   */
  void Train(bool resume, const std::string str); 
  /**
   * Create workers and servers to conduct the training.
   *
   * @param[in] job_conf job configuration with all necessary fields set (e.g.,
   * by Train(bool, const JobProto&).
   */
  void Train(const JobProto& job_conf);
  /**
   * Test the pre-trained model by loading parameters from checkpoint files.
   *
   * It can be used for both computing accuracy of test data, and extracting
   * features (predicting label) of new data.
   * @param[in] job_conf job configuration, which should include the checkpoint
   * files and test settings (e.g., test steps). To extract features, the output
   * layers should be added.
   */
  void Test(const JobProto& job_conf);
  /**
   * Used for python binding. Users can also directly call it as a C++ API.
   *
   * It completes the functions as defined above but accept serialized string
   * parameters.
   *
   * @param[in] str serialized string recorded job configuration.
   */
  void Test(const std::string str);
  /**
   * Setting the checkpoint field of the job configuration to resume training.
   *
   * The checkpoint folder will be searched to get the files for the latest
   * checkpoint, which will be added into the checkpoint field. The workers
   * would then load the values of params from the checkpoint files.
   *
   * @param job_conf job configuration
   */
  void SetupForResume(JobProto* job_conf);
  /**
   * Create server instances.
   *
   * @param[in] job_conf job configuration.
   * @param[in] net training neural network.
   * @return server instances
   */
  const vector<Server*> CreateServers(const JobProto& job_conf, NeuralNet* net);
  /**
   * Create workers instances.
   * @param[in] job_conf job configuration.
   * @param[in] net training neural network.
   * @return worker instances
   */
  const vector<Worker*> CreateWorkers(const JobProto& job_conf, NeuralNet* net);


  /*********** Subclasses registers *************************/
  /**
   * Register a Layer subclass.
   *
   * @param type layer type ID. If called to register built-in subclasses,
   * it is from LayerType; if called to register user-defined
   * subclass, it is a string;
   * @return 0 if success; otherwise -1.
   */
  template<typename Subclass, typename Type>
  int RegisterLayer(const Type& type);
  /**
   * Register an Updater subclass.
   *
   * @param type ID of the subclass. If called to register built-in subclasses,
   * it is from UpdaterType; if called to register user-defined
   * subclass, it is a string;
   * @return 0 if success; otherwise -1.
   */
  template<typename Subclass, typename Type>
  int RegisterUpdater(const Type& type);
  /**
   * Register a learning rate generator subclasses.
   *
   * @param type ID of the subclass. If called to register built-in subclasses,
   * it is from ChangeMethod; if called to register user-defined
   * subclass, it is a string;
   * @return 0 if success; otherwise -1.
   */
  template<typename Subclass, typename Type>
  int RegisterLRGenerator(const Type& type);
  /**
   * Register a Worker subclass.
   *
   * @param type ID of the subclass. If called to register built-in subclasses,
   * it is from TrainOneBatchAlg; if called to register user-defined
   * subclass, it is a string;
   * @return 0 if success; otherwise -1.
   */
  template<typename Subclass, typename Type>
  int RegisterWorker(const Type& type);
  /**
   * Register a Param subclass.
   * @param type ID of the subclass. If called to register built-in subclasses,
   * it is from ParamType; if called to register user-defined
   * subclass, it is a string;
   *
   * @return 0 if success; otherwise -1.
   */
  template<typename Subclass, typename Type>
  int RegisterParam(const Type& type);
  /**
   * Register ParamGenerator subclasses for initalizing Param objects.
   *
   * @param type ID of the subclass. If called to register built-in subclasses,
   * it is from InitMethod; if called to register user-defined
   * subclass, it is a string;
   * @return 0 if success; otherwise -1.
   */
  template<typename Subclass, typename Type>
  int RegisterParamGenerator(const Type& type);

  /****************** Access function ********************/
  /**
   * @return job ID which is generated by zookeeper and passed in by the
   * launching script.
   */
  inline int job_id() const { return job_id_; }
  /**
   * @return job conf path which is passed by users at the command line. It
   * should at least contains the cluster configuration.
   */
  inline JobProto job_conf() const { return job_conf_; }

 private:
  int job_id_;
  JobProto job_conf_;
  SingaProto singa_conf_;
};

/************* Implementation of template functions*************************
* Must put the implementation in driver.h file instead of driver.cc.
* Otherwise there would be linking error caused by unknown registration
* functions, becuase these function cannot be generated merely based on its
* declearation in driver.h.
*/

template<typename Subclass, typename Type>
int Driver::RegisterLayer(const Type& type) {
  auto factory = Singleton<Factory<singa::Layer>>::Instance();
  factory->Register(type, CreateInstance(Subclass, Layer));
  return 1;
}

template<typename Subclass, typename Type>
int Driver::RegisterParam(const Type& type) {
  auto factory = Singleton<Factory<singa::Param>>::Instance();
  factory->Register(type, CreateInstance(Subclass, Param));
  return 1;
}

template<typename Subclass, typename Type>
int Driver::RegisterParamGenerator(const Type& type) {
  auto factory = Singleton<Factory<singa::ParamGenerator>>::Instance();
  factory->Register(type, CreateInstance(Subclass, ParamGenerator));
  return 1;
}

template<typename Subclass, typename Type>
int Driver::RegisterUpdater(const Type& type) {
  auto factory = Singleton<Factory<singa::Updater>>::Instance();
  factory->Register(type, CreateInstance(Subclass, Updater));
  return 1;
}

template<typename Subclass, typename Type>
int Driver::RegisterLRGenerator(const Type& type) {
  auto factory = Singleton<Factory<singa::LRGenerator>>::Instance();
  factory->Register(type, CreateInstance(Subclass, LRGenerator));
  return 1;
}

template<typename Subclass, typename Type>
int Driver::RegisterWorker(const Type& type) {
  auto factory = Singleton<Factory<singa::Worker>>::Instance();
  factory->Register(type, CreateInstance(Subclass, Worker));
  return 1;
}

}  // namespace singa

#endif  // SINGA_SINGA_DRIVER_H_
