#ifndef INCLUDE_TRAINER_TRAINER_H_
#define INCLUDE_TRAINER_TRAINER_H_
#include <unordered_map>
#include "proto/cluster.pb.h"
#include "proto/model.pb.h"
#include "utils/updater.h"
#include "utils/param.h"
#include "utils/singleton.h"
#include "utils/factory.h"
#include "neuralnet/neuralnet.h"
#include "trainer/worker.h"
#include "trainer/server.h"
#include "communication/socket.h"

namespace singa {
/**
 * Callback function for zookeeper
 */
void HandleWorkerFinish(void * ctx);
/**
 * Zookeeper handler context used by HandleWorkerFinish(void*)function.
 */
typedef struct HandleContext_{
  shared_ptr<Dealer> dealer;
  int group_id, id;
} HandleContext;
/**
  * ParamInfo is used to construct a parameter shard.
  *
  * For each worker group:
  *   Every unique Param object is associated with a ParamCounter object whose
  *   param field points the to Param object itself.
  *
  *   Param objects sharing the same values (due to data parallelism) are
  *   associated with the same ParamCounter whose param field also shares the
  *   same values.
  *
  *   Usage: we need to aggregate gradients from all workers for the shared
  *   parameters before sending the update request. The nUpdate counter counts
  *   the number.
  *
  * TODO test with different physical architectures.
  */
class ParamInfo{
   public:
  ParamInfo(shared_ptr<Param> p,int local, int owner):
    num_update(0), next_version(0),num_local(local), num_total(1),
    owner_procs(owner){
      shares.push_back(p);
    }

  /**
    * Associate the counter to a Param object.
    *
    * @param p
    * @param local 1 if this Param object is used by workers in this procs, 0
    *  otherwise
    * @param owner the procs id of the worker who ownes this Param object
    */
  void AddParam(shared_ptr<Param> p, bool local){
    num_local+=local;
    num_total+=1;
    if(local)
      shares.push_back(p);
  }
  int num_update, next_version; //!< all counters are atomic

  int num_local; //!< # local workers uses the shared parameter
  int num_total; //!< # total workers uses the shared parameter
  int owner_procs; //!< the procs id of the worker that owns the parameter
  vector<shared_ptr<Param>> shares;
};

typedef std::map<int, shared_ptr<ParamInfo>> WorkerShard;

/**
 * Every running process has a training object which launches one or more
 * worker (and server) threads.
 *
 * The main thread runs a loop to forward messages between workers and servers.
 */

class Trainer{
 public:
  /**
   * Start the training in one process
   *
   * @param modelproto
   * @param clusterproto
   */
  void Start(const ModelProto& modelproto, const ClusterProto& clusterproto,
    const int procs_id);

  // TODO add Resume() function to continue training from a previously stopped
  // point.

 protected:
  vector<shared_ptr<Server>> CreateServers(int nthread, const ModelProto& mproto,
      const vector<int> slices, vector<HandleContext*>* ctx);
  vector<shared_ptr<Worker>> CreateWorkers(int nthread,
      const ModelProto& mproto, vector<int> *slice_size);

  void Run(const vector<shared_ptr<Worker>>& workers,
      const vector<shared_ptr<Server>>& servers,
      const std::map<int, shared_ptr<ParamShard>>& shards);
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
   * Workers from the same group resident in the same process share the same
   * WorkerShard which contains ParamCounters for Param objects used/updated by
   * these worekrs. Shared Param objects are associated with the same
   * ParamCounter.
   */

	/**
	 * Generate a request message to Get the parameter object.
	 */
	virtual const vector<Msg*> HandleGet(shared_ptr<ParamInfo>counter, Msg** msg);
	virtual void HandleGetResponse(shared_ptr<ParamInfo>counter, Msg** msg);

	/**
	 * Generate a request message to Update the parameter object.
	 */
	virtual const vector<Msg*> HandleUpdate(shared_ptr<ParamInfo>counter, Msg** msg);
  virtual void HandleUpdateResponse(shared_ptr<ParamInfo>counter, Msg** msg);

  /**
	 * Generate a request message to Put the parameter object.
	 */
	virtual const vector<Msg*> HandlePut(shared_ptr<ParamInfo>counter, Msg** msg);
	virtual Msg* HandleConnect(Msg** msg);

 protected:
  int procs_id_;
  shared_ptr<Router> router_;
  std::unordered_map<int, shared_ptr<WorkerShard>> worker_shards_;
  shared_ptr<ServerShard> server_shard_;
  vector<int> slice2server_;
};
} /* singa */
#endif // INCLUDE_TRAINER_TRAINER_H_
