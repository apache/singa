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

#ifndef SINGA_WORKER_H_
#define SINGA_WORKER_H_

#include <string>
#include <unordered_map>
#include <vector>
#include "singa/comm/socket.h"
#include "singa/neuralnet/neuralnet.h"
#include "singa/proto/job.pb.h"
#include "singa/neuralnet/connection_layer.h"
#include "singa/neuralnet/neuron_layer.h"

namespace singa {

//!< sleep 5 milliseconds if the Param is not updated to the expected version
const int kCollectSleepTime = 5;
/**
 * The Worker class which runs the training algorithm.
 * The first worker group will initialize parameters of the Net,
 * and put them into the distributed memory/table.
 * The virtual function TrainOneBatch and TestOneBatch implement the
 * training and test algorithm for one mini-batch data.
 *
 * Child workers override the two functions to implement their training
 * algorithms, e.g., the BPWorker/CDWorker/BPTTWorker implements the BP/CD/BPTT
 * algorithm respectively.
 */
class Worker {
 public:
  /**
   * Create an instance of the subclass of Worker.
   *
   * @param[in] conf configuration of the TrainOneBatch algorithm. Different
   * Worker subclasses implement different algorithms. Hence the creation is
   * based on the TrainOneBatch algorithm type. Currently SINGA
   * provides two algorithms:
   * -# Back-propagation for the feed-forward models, e.g., CNN and MLP, and the
   *  recurrent neural networks.
   * -# Contrastive divergence for the energy models, e.g., RBM.
   *
   * @return a pointer to the instance of the Worker subclass.
   */
  static Worker* CreateWorker(const std::string str);
  static Worker* Create(const AlgProto& conf);
  virtual ~Worker();
  /**
   * @param[in] grp_id global worker group ID
   * @param[in] id worker ID within the group
   * @param[in] conf job configuration
   * @param[in] train_net pointer to the training neural net, which could be
   * shared with other workers from the same group. Different workers run over
   * differnt subset of layers.
   * @param[in] val_net pointer to the validation neural net. Currently only the
   * first worker from the first group would have validation neural net. All
   * other workers receive nullptr for this argument.
   * @param[in] test_net pointer to the test neural net. Currently only the
   * first worker from the first group would have test neural net. All other
   * workers receive nullptr for this argument.
   */
  virtual void Setup(int grp_id, int id, const JobProto& conf,
      NeuralNet* train_net, NeuralNet* val_net, NeuralNet* test_net);
  /**
   * Main function of Worker.
   *
   * Train the neuralnet step by step, test/validation is done periodically.
   */
  void Run();
  /**
   * Run TestOneBatch() over the a neural net for a total number of steps.
   *
   * @param[in] steps total number of test iterations.
   * @param[in] phase kVal or kTest
   * @param[in] net run test over the passed in neural net
   */
  void Test(int steps, Phase phase, NeuralNet* net);
  /**
   * Init sockets in a worker, including:
   * 1. a global socket communicates with stub
   * 2. a bridge socket dedicated for bridge layer communications
   *
   * the bridge socket will be binded to each bridge layer
   *
   * @param[in] net pointer to a neural net whose bridge layer will be binded
   * with a socket.
   */
  void InitSockets(const NeuralNet* net);
  /**
   * Init values of Param instances assocaited with local layers (i.e., layers
   * dispatched to this worker).
   *
   * If one Param is owned by the worker, then it should be initialized and put
   * to servers. Otherwise Get() should be called to get the Param. The Get()
   * may not send get requests if the Param owner is in the same procs, for
   * which case the memory space of the Param objects are shared. But if this
   * worker and the Param owner worker run on different devices (e.g., GPUs),
   * then the get request would be sent.
   *
   * If the training starts from scrath, every Param object is initialzed using
   * ParamGenerator. After that, the worker may
   * train for a couple of steps to warmup the params before put
   * them to servers (warmup of JobProto controls this).
   *
   * If one Param object's name matches that of one Param object from the
   * checkpoint files, its Param values would be loaded from checkpoint files.
   *
   * @param[in] job_conf job configuration which provides settings for
   * checkpoint file paths, warmup steps and Param versions.
   * @param[out] net pointer to a neural net whose Param values will be
   * initialized.
   */
  void InitNetParams(const JobProto& job_conf, NeuralNet* net);
  void InitNetParams(const std::string& folder, vector<Layer*> net);
  /**
   * Checkpoint all Param objects owned by the worker onto disk.
   * The serialization is done using BlobProtos which includes the name, version
   * and values of each Param object.
   * Different workers would generate different checkpoint files. The file path
   * is <workspace>/checkpoint-<jobname>-step<step>-worker<worker_id>.bin
   * @param[in] step training step
   * @param[in] folder directory to put the checkpoint file
   * @param net the training net whose Param objects will be dumped.
   */
  void Checkpoint(int step, const std::string& folder, NeuralNet* net);
  void Checkpoint(int step, const std::string& folder, vector<Layer*> net);
  /**
    * Train one mini-batch.
    * Test/Validation is done before training.
    *
    * @param[in] step training step.
    * @param[in] net neural net to be trained.
    */
  virtual void TrainOneBatch(int step, NeuralNet* net) = 0;
  /**
   * Test/validate one mini-batch data.
   *
   * @param[in] step test step.
   * @param[in] phase test could be done for validation or test phase.
   * @param[in] net neural net for test
   */
  virtual void TestOneBatch(int step, Phase phase, NeuralNet* net) = 0;
  /**
   * Display infomation from layers.
   *
   * @param flag could be a combination of multiple phases, e.g, kTest|kForward,
   * it is passed to the Layer::ToString() function for each layer to decide
   * what to display .
   * @param prefix display prefix, e.g., 'Train step 100', 'Test step 90'.
   * @param net display layers from this neural net.
   */
  virtual void Display(int flag, const std::string& prefix, NeuralNet* net);
  /**
   * Put Param values to server.
   *
   * @param param
   * @param step used as current param version for the put request
   */
  int Put(int step, Param* param);
  /**
   * Get Param with specific version from server
   * If the current version >= the requested version, then return.
   * Otherwise send a get request to stub who would forwards it to servers.
   * @param param
   * @param step requested param version
   */
  int Get(int step, Param* param);
  /**
   * Update Param.
   *
   * @param param
   * @param step training step used for updating (e.g., deciding learning rate).
   */
  int Update(int step, Param* param);
  /**
   * Wait for the response of the update/get requests.
   *
   * @param param
   * @param step not used now.
   */
  int Collect(int step, Param* param);
  /**
   * Call Collect() for every param of net
   */
  int CollectAll(int step, NeuralNet* net);
  /**
   * @param[in] step
   * @return true if it is time to display training info, e.g., loss; otherwise
   * false.
   */
  inline bool DisplayNow(int step) const {
    return job_conf_.disp_freq() > 0
           && step >= job_conf_.disp_after()
           && ((step - job_conf_.disp_after()) % job_conf_.disp_freq() == 0);
  }
  /**
   * @param[in] step
   * @return true if it is time to finish the training; otherwise false.
   */
  inline bool StopNow(int step) const {
    return step >= job_conf_.train_steps();
  }
  /**
   * @param[in] step
   * @return true if it is time to do checkpoint Param objects; otherwise false.
   */
  inline bool CheckpointNow(int step) const {
    return job_conf_.checkpoint_freq() > 0
           && step >= job_conf_.checkpoint_after()
           && ((step - job_conf_.checkpoint_after())
              % job_conf_.checkpoint_freq() == 0);
  }
  /**
   * @param[in] step
   * @return true if it is time to do test over the test dataset.
   */
  inline bool TestNow(int step) const {
    return job_conf_.test_freq() > 0
      && job_conf_.test_steps() > 0
      && step >= job_conf_.test_after()
      && ((step - job_conf_.test_after()) % job_conf_.test_freq() == 0);
  }
  /**
   * @param[in] step
   * @return true if it is time to do test over the validation dataset.
   */
  inline bool ValidateNow(int step) const {
    return job_conf_.validate_freq() > 0
      && job_conf_.validate_steps() > 0
      && step >= job_conf_.validate_after()
      && ((step - job_conf_.validate_after()) % job_conf_.validate_freq() == 0);
  }
  /**
   * @return a vector with pointers to all neural nets.
   */
  const std::vector<NeuralNet*> GetNets() const {
    return std::vector<NeuralNet*> {train_net_, val_net_, test_net_};
  }
  /**
   * @return training net.
   */
  inline NeuralNet* train_net() const {
    return train_net_;
  }
  /**
   * @return group ID
   */
  inline int grp_id() const { return grp_id_; }
  /**
   * @reutrn worker ID within the worker group.
   */
  inline int id() const { return id_; }

 protected:
  int grp_id_ = -1, id_ = -1;
  int step_ = 0;
  JobProto job_conf_;
  NeuralNet* train_net_ = nullptr;
  NeuralNet* test_net_ = nullptr;
  NeuralNet* val_net_ = nullptr;
  Dealer* dealer_ = nullptr;
  // bridge layer related
  Dealer* bridge_dealer_ = nullptr;
  std::unordered_map<std::string, Layer*> name2bridge_;
};

class BPWorker: public Worker {
 public:
  void TrainOneBatch(int step, NeuralNet* net) override;
  void TestOneBatch(int step, Phase phase, NeuralNet* net) override;
  virtual void Forward(int step, Phase phase, NeuralNet* net);
  virtual void Backward(int step, NeuralNet* net);
};

/**
 * Subclass of Worker that implements BPTT (Backpropagation through time)
 * algorithm for computing gradients of RNN models.
 * Max BPTT/unrolling length is configured by users.
 */
class BPTTWorker: public BPWorker {
 public:
  void Forward(int step, Phase phase, NeuralNet* net) override;
  void Backward(int step, NeuralNet* net) override;
  void Display(int flag, const std::string& prefix, NeuralNet* net) override;

 private:
  /*
   * indicator used in truncted BPTT, which feeds the hidden state of the last
   * unrolled unit to the first unit in Forward() for the next iteration.
   * currently always feed the last hidden state to the first.
   */
  bool full_state_ = false;
  //!< indicator used for the starting of a new pass of the dataset.
  bool begin_ = false;
};
/**
 * Subclass of Worker that implements the Contrastive Divergence algorithm for
 * computing the gradients of paramters of energy models.
 */
class CDWorker: public Worker {
 public:
  void TrainOneBatch(int step, NeuralNet* net) override;
  void TestOneBatch(int step, Phase phase, NeuralNet* net) override;
};

inline int BlobTrgt(int grp, int layer) {
  return (grp << 16) | layer;
}

inline int BlobGrp(int blob_trgt) {
  return blob_trgt >> 16;
}

inline int BlobLayer(int blob_trgt) {
  static int mask = (1 << 16) -1;
  return blob_trgt & mask;
}

}  // namespace singa

#endif  // SINGA_WORKER_H_
