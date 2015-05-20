#ifndef INCLUDE_TRAINER_WORKER_H_
#define INCLUDE_TRAINER_WORKER_H_
#include <map>
#include <exception>
#include "neuralnet/neuralnet.h"
#include "proto/model.pb.h"
#include "utils/cluster.h"
#include "communication/socket.h"
#include "communication/msg.h"

namespace singa {
const int kCollectSleepTime=5;//milliseconds;
/**
 * The Worker class which runs the training algorithm.
 * The first worker group will initialize parameters of the Net,
 * and put them into the distributed memory/table.
 */
class Worker {
 public:
  Worker(int thread_id, int group_id, int worker_id);
  ~Worker(){}
  void Setup(const ModelProto& model, shared_ptr<NeuralNet> train_net);
  void set_test_net(shared_ptr<NeuralNet> test_net){
    test_net_=test_net;
  }
  void set_validation_net(shared_ptr<NeuralNet> val_net){
    validation_net_=val_net;
  }

  int Put(shared_ptr<Param> param, int step);
  int Get(shared_ptr<Param> param, int step);
  int Update(shared_ptr<Param> param, int step);
  int Collect(shared_ptr<Param> param, int step);
  int CollectAll(shared_ptr<NeuralNet> net, int step);
  /**
    * check validation/test firstly, then TrainOneBatch
    * Performance collects performance for the whole neuralnet.
    * Hence, no need to collect performance in every thread.
    * Only the main thread will pass none null perf.
    */
  void RunOneBatch(int step, Metric* perf=nullptr);
  /**
    * Train one mini-batch.
    * Test/Validation is done before training.
    */
  virtual void TrainOneBatch(int step)=0;
  /**
   * Test/validate one mini-batch.
   */
  virtual void TestOneBatch(shared_ptr<NeuralNet> net, int step, Phase phase)=0;
  /**
    * Test the perforance of the learned model on validation or test dataset.
    * Test is done by the first group.
    * @param net, neural network
    * @param phase kValidation or kTest.
    */
  void Test(shared_ptr<NeuralNet> net, int nsteps, bool dispperf);

  /**
    * Main function of Worker.
    * 1. Train the neuralnet step by step, test/validation is done periodically.
    * 2. TODO Communicate with others, e.g., zookeeper, after every step.
    */
  virtual void Run();


  /**
   * Pull data from layers resident on other nodes due to Model Partition.
  void Pull(zsock_t* pull, shared_ptr<NeuralNet> net);
   */

  /**
   * Check is it time to display training info, e.g., loss and precison.
   */
  const bool DisplayNow(const int step) const {
    return (modelproto_.display_frequency() > 0
        && step >= modelproto_.display_after_steps()
        && ((step - modelproto_.display_after_steps())
          % modelproto_.display_frequency() == 0));
  }

  const bool DisplayDebugInfo(const int step) const {
    return DisplayNow(step)&&modelproto_.debug()&&group_id_==0;
  }
  const void DisplayPerformance(const Metric & perf, const string& prefix);

  /**
   * return true if the stop condition is satisfied, e.g., the maximum number
   * of steps have been reached.
   */
  const bool StopNow(const int step) const{
    return (step >= modelproto_.train_steps());
  }
  /**
   * Check is it time to do checkpoint.
   * @param step the ::Train() has been called this num times.
   */
  const bool CheckpointNow(const int step) const{
    return (group_id_==0
        && modelproto_.checkpoint_frequency() > 0
        && step >= modelproto_.checkpoint_after_steps()
        && ((step - modelproto_.checkpoint_after_steps())
          % modelproto_.checkpoint_frequency() == 0));
  }
  /**
   * Check is it time to do test.
   * @param step the ::Train() has been called this num times.
   */
  const bool TestNow(const int step) const{
    return (group_id_==0
        && modelproto_.test_frequency() > 0
        && step >= modelproto_.test_after_steps()
        && ((step - modelproto_.test_after_steps())
          % modelproto_.test_frequency() == 0));
  }
  /**
   * Check is it time to do validation.
   * @param step the ::Train() has been called step times.
   */
  const bool ValidateNow(const int step) {
    return (group_id_==0
        && modelproto_.validation_frequency() > 0
        && step >= modelproto_.validation_after_steps()
        && ((step - modelproto_.validation_after_steps())
          % modelproto_.validation_frequency() == 0));
  }


  /**
   * start training from scratch.
   * setup training/test/validation neuralnets, then call Run().
  void Start(ModelProto model);
   */
  /**
   * TODO Resume from snapshot
  void Resume();
   */
  void ReceiveBlobs(shared_ptr<NeuralNet> net);
  void SendBlob();
 protected:
  int thread_id_, group_id_, worker_id_;
  int step_;
  ModelProto modelproto_;
  shared_ptr<NeuralNet> train_net_, test_net_, validation_net_;
  shared_ptr<Dealer> layer_dealer_, param_dealer_;
  Poller layer_poller_, param_poller_;
};

class BPWorker: public Worker{
 public:
  ~BPWorker(){}
  BPWorker(int thread_id, int group_id, int worker_id):Worker(thread_id, group_id, worker_id){}
  virtual void TrainOneBatch(int step);
  virtual void TestOneBatch(shared_ptr<NeuralNet> net, int step, Phase phase);
  void Forward(shared_ptr<NeuralNet> net, int step, bool training);
  void Backward(shared_ptr<NeuralNet> net, int step);
    /**
   * Profiling the time cost of training one batch.
  string TimerInfo(){
    char buf[1024];
    float ticks=ticks_*1000;
    float tf=tForward_/ticks, tb=tBackward_/ticks,
          td=tSyncData_/ticks, tp=tSyncParam_/ticks;
    float total=tf+tb+td+tp;
    sprintf(buf,
        "Total\t%6.2f\tforward\t%6.2f\tbackward\t%6.2f\t"
        // syncdata\t%6.2f\tsyncparam\t%6.2f\n"
        , total,tf,tb);
    float gensync=Param::worker_gen_sync/ticks;
    float handlesync=Param::worker_handle_sync/ticks;
    sprintf(buf+strlen(buf),
        "worker_gen_sync\t%6.2f\tworker_handle_sync\t%6.2f\n",
        gensync, handlesync);
    Param::worker_gen_sync=0;
    Param::worker_handle_sync=0;
    tForward_=0;
    tBackward_=0;
    tSyncData_=0;
    tSyncData_=0;
    ticks_=0;
    return string(buf);
  }
   */
};
}  // namespace singa

#endif  // INCLUDE_TRAINER_WORKER_H_
