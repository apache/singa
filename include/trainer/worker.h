#ifndef SINGA_TRAINER_WORKER_H_
#define SINGA_TRAINER_WORKER_H_
#include "neuralnet/neuralnet.h"
#include "proto/job.pb.h"
#include "utils/updater.h"
#include "communication/socket.h"

namespace singa {
//!< sleep 5 milliseconds if the Param is not updated to the expected version
const int kCollectSleepTime=5;
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
   * @param thread_id local thread index within the procs
   * @param grp_id global worker group ID
   * @param id worker ID within the group
   */
  Worker(int thread_id, int grp_id, int id);
  virtual ~Worker();
  /**
   * Setup members
   */
  void Setup(const ModelProto& model, shared_ptr<NeuralNet> train_net,
      shared_ptr<NeuralNet> valid_net, shared_ptr<NeuralNet> test_net);
  /**
    * Main function of Worker.
    *
    * Train the neuralnet step by step, test/validation is done periodically.
    */
  void Run();
  /**
   * Init all local params (i.e., params from layers resident in this worker).
   *
   * If the param is owned by the worker, then init it and put it to servers.
   * Otherwise call Get() to get the param. The Get may not send get request.
   * Because the param's own is in the same procs. Once the owner initializes
   * the param, its version is visiable to all shares.
   * If the training starts from scrath, the params are initialzed using random
   * distributions, e.g., Gaussian distribution. After that, the worker may
   * train for a couple of steps to warmup the params before put
   * them to servers (warmup of ModelProto controls this).
   *
   * If the owner param is availabel from checkpoint file, then its
   * values are parsed from the checkpoint file instead of randomly initialized.
   * For params who do not have checkpoints, randomly init them.
   */
  void InitLocalParams();

  /**
   * Checkpoint all params owned by the worker from the first group onto disk.
   * The serialization is done using BlobProtos which includes the name, version
   * and values of each Param.
   * Different worker would generate different checkpoint files. The file path
   * is <workspace>/checkpoint-<modelname>-step<step>-worker<worker_id>.bin
   * @param step training step of this worker
   * @param net the training net whose params will be dumped.
   */
  void Checkpoint(int step, shared_ptr<NeuralNet> net);
  /**
    * Test the perforance of the learned model on validation or test dataset.
    * Test is done by the first group.
    * @param net, neural network
    */
  void Test(int nsteps, Phase phase, shared_ptr<NeuralNet> net);
  /**
    * Train one mini-batch.
    * Test/Validation is done before training.
    */
  virtual void TrainOneBatch(int step, Metric* perf)=0;
  /**
   * Test/validate one mini-batch.
   */
  virtual void TestOneBatch(int step, Phase phase, shared_ptr<NeuralNet> net,
      Metric* perf)=0;
  /**
   * Report performance to the stub.
   *
   * @param prefix display prefix, e.g., 'Train', 'Test'
   * @param perf
   */
  void Report(const string& prefix, const Metric & perf);

  /**
   * Put Param to server.
   * @param param
   * @param step used as current param version for the put request
   */
  int Put(Param* param, int step);
  /**
   * Get Param with specific version from server
   * If the current version >= the requested version, then return.
   * Otherwise send a get request to stub who would forwards it to servers.
   * @param param
   * @param step requested param version
   */
  int Get(Param* param, int step);
  /**
   * Update Param
   * @param param
   * @param step training step used for updating (e.g., deciding learning rate)
   */
  int Update(Param* param, int step);
  /**
   * Block until the param is updated since sending the update request
   *
   * @param param
   * @param step not used
   */
  int Collect(Param* param, int step);
  /**
   * Call Collect for every param of net
   */
  int CollectAll(shared_ptr<NeuralNet> net, int step);
  /**
   * Receive blobs from other workers due to model partitions.
   */
  void ReceiveBlobs(
    bool data, bool grad, BridgeLayer* layer, shared_ptr<NeuralNet> net);
  /**
   * Send blobs to other workers due to model partitions.
   */
  void SendBlobs(
    bool data, bool grad, BridgeLayer* layer, shared_ptr<NeuralNet> net);

  /**
   * Check is it time to display training info, e.g., loss and precison.
   */
  inline bool DisplayNow(int step) const;
  /**
   * Check is it time to display training info, e.g., loss and precison.
   */
  inline bool DisplayDebugInfo(int step) const;
  /**
   * Check is it time to stop
   */
  inline bool StopNow(int step) const;
  /**
   * Check is it time to do checkpoint.
   */
  inline bool CheckpointNow(int step) const;
  /**
   * Check is it time to do test.
   * @param step the ::Train() has been called this num times.
   */
  inline bool TestNow(int step) const;
  /**
   * Check is it time to do validation.
   * @param step the ::Train() has been called step times.
   */
  inline bool ValidateNow(int step) const;

  /**
   * @return group ID
   */
  int grp_id() const { return grp_id_;}

  /**
   * @reutrn worker ID within the worker group.
   */
  int id() const { return id_;}

 protected:
  int thread_id_, grp_id_, id_;
  int step_;
  ModelProto modelproto_;
  shared_ptr<NeuralNet> train_net_, test_net_, validation_net_;
  Dealer* layer_dealer_, *dealer_;
  Updater* updater_;
};

class BPWorker: public Worker{
 public:
  BPWorker(int thread_id, int grp_id, int id);
  ~BPWorker(){}
  void TrainOneBatch(int step, Metric* perf) override;
  void TestOneBatch(int step, Phase phase, shared_ptr<NeuralNet> net,
      Metric* perf) override;

  void Forward(int step, Phase phase, shared_ptr<NeuralNet> net, Metric* perf);
  void Backward(int step, shared_ptr<NeuralNet> net);
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

#endif  // SINGA_TRAINER_WORKER_H_
