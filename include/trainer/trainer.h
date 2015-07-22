#ifndef INCLUDE_TRAINER_TRAINER_H_
#define INCLUDE_TRAINER_TRAINER_H_
#include <unordered_map>
#include <queue>
#include "proto/job.pb.h"
#include "proto/singa.pb.h"
#include "utils/param.h"
#include "utils/singleton.h"
#include "utils/factory.h"
#include "neuralnet/neuralnet.h"
#include "trainer/worker.h"
#include "trainer/server.h"
#include "communication/socket.h"

namespace singa {
/**
 * Every running process has a training object which launches one or more
 * worker (and server) threads.
 *
 * The main thread runs a loop to forward messages between workers and servers.
 */

class Trainer{
 public:
  ~Trainer();
  /**
   * Entrance function which construct the workers and servers, and luanch
   * one thread per worker/server.
   *
   * @param job job ID
   * @param resume if true resume the training from the latest checkpoint files
   * @param jobConf job configuration, including cluster and model configuration
   * @param singaConf global singa configuration including zookeeper and
   * log dir setting.
   */
  void Start(int job, bool resume,
      const JobProto& jobConf, const SingaProto& singaConf);

 protected:
  /**
   * Setting the checkpoint field of model configuration to resume training.
   *
   * The checkpoint folder will be searched to get the files for the latest
   * checkpoint, which will be added into the checkpoint field. The workers
   * would then load the values of params from the checkpoint files.
   *
   * @param modelConf model configuration
   */
  void Resume(ModelProto* modelConf);
  /**
   * Create server instances.
   * @param nthread total num of threads in current procs which is used to
   * assign each thread a local thread ID. The number of workers is extracted
   * from Cluster
   * @param modelConf
   * @return server instances
   */
  vector<Server*> CreateServers(int nthread, const ModelProto& modelConf);
  /**
   * Create workers instances.
   * @param nthread total num of threads in current procs which is used to
   * assign each thread a local thread ID. The number of workers is extracted
   * from Cluster
   * @param modelConf
   * @return worker instances
   */
  vector<Worker*> CreateWorkers(int nthread, const ModelProto& modelConf);

  /**
   * Setup workers and servers.
   *
   * For each worker, create and assign a neuralnet to it.
   * For each server, create and assign the param shard to it.
   * Create the partition map from slice ID to server
   * @param modelConf
   * @param workers
   * @param servers
   */
  void SetupWorkerServer(
    const ModelProto& modelConf,
    const vector<Worker*>& workers,
    const vector<Server*>& servers);

  void Run(const vector<Worker*>& workers, const vector<Server*>& servers);
  /**
   * Register default implementations for all base classes used in the system,
   * e.g., the Updater, BaseMsg, etc.
   *
   * All built-in layer implementations are
   * registered here.
   * For other base classes, use its base class name (string) as the key and the
   * implementation class as the value, e.g., <"Updater" SGDUpdater>.
   */
  void RegisterDefaultClasses(const singa::ModelProto& proto);
  /**
   * Generate msg to trigger synchronization with other server groups.
   *
   * @param server the local server index whom the message is sent to
   * @param servers all local servers
   * @return sync msg
   */
  Msg* GenSyncReminderMsg(int server, const vector<Server*>& servers);
  /**
   * Display metrics to log (standard output)
   */
  void DisplayMetric(Msg** msg);
  /**
   * Create a socket to send msg to the specified process
   * @param dst_procs the dst process (logical) ID
   * @return the newly created socket
   */
  Dealer* CreateInterProcsDealer(int dst_procs);
  /**
   * Handle messages to local servers and local stub
   */
  void HandleLocalMsg(std::queue<Msg*>* msg_queue, Msg** msg);

	/**
	 * Generate a request message to Get the parameter object.
	 */
	const vector<Msg*> HandleGet(ParamEntry* entry, Msg** msg);
	void HandleGetResponse(ParamEntry* entry, Msg** msg);

	/**
	 * Generate a request message to Update the parameter object.
	 */
	const vector<Msg*> HandleUpdate(ParamEntry* entry, Msg** msg);
  void HandleUpdateResponse(ParamEntry* entry, Msg** msg);

  /**
	 * Generate a request message to Put the parameter object.
	 */
	const vector<Msg*> HandlePut(ParamEntry* entry, Msg** msg);

  /**
   * Called by HandlePut, HandleUpdate and HandleGet functions
   * @param type message type
   * @param version param version
   * @param entry
   * @param msg
   * @param ret generated messages
   */
  void GenMsgs(int type, int version, ParamEntry* entry,
    Msg* msg, vector<Msg*> *ret);
  /**
   * Get a hash id for a Param object from a group.
   *
   * Simple multiple group_id with a large prime number 997 (assuming there are
   * no more than 997 worker groups) and plus owner param id.
   */
  inline int Hash(int grp_id, int param_id) {
    return grp_id * 997 + param_id;
  }

 protected:
  int procs_id_;
  Router *router_;
  std::unordered_map<int, ParamEntry*> worker_shard_;
  //!< map from slice ID to slice, used by servers and deleted in the destructor
  std::unordered_map<int, ParamEntry*> server_shard_;
  //!< map from slice to the server that updates it
  vector<int> slice2server_;
};
} /* singa */
#endif // INCLUDE_TRAINER_TRAINER_H_
