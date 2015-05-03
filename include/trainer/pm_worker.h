#ifndef INCLUDE_TRAINER_PM_WORKER_H_
#define INCLUDE_TRAINER_PM_WORKER_H_

#include <memory>
#include <vector>
#include <map>
#include <string>
#include <atomic>
#include "utils/param.h"
#include "communication/msg.h"

using std::string;
using std::vector;
using std::shared_ptr;
using std::map;

namespace singa {

/**
 * Counters used to construct a parameter shard.
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
class ParamCounter{
  public:
  ParamCounter(shared_ptr<Param> p,int local, int owner):
    nUpdate(0), nGet(0), nPut(0), nCollect(0), nLocal(local), nTotal(0),
    owner_procs(owner), param(p){}

  /**
   * Associate the counter to a Param object.
   *
   * @param p
   * @param local 1 if this Param object is used by workers in this procs, 0
   *  otherwise
   * @param owner the procs id of the worker who ownes this Param object
   */
  void AddParam(shared_ptr<Param> p, int local, int owner){
    nLocal+=local;
    nTotal+=1;
    if(owner_procs>-1)
      owner_procs=owner;
    if(nLocal>1){
      // TODO copy p->param;
    }
  }
  std::atomic<int> nUpdate, nGet, nPut, nCollect; //!< all counters are atomic

  int nLocal; //!< # local workers uses the shared parameter
  int nTotal; //!< # total workers uses the shared parameter
  int owner_procs; //!< the procs id of the worker that owns the parameter
  shared_ptr<Param> param;
};


/**
 * Parameter manager at the worker side.
 */
class PMWorker{
public:
  /**
   * Workers from the same group resident in the same process share the same
   * ParamShard which contains ParamCounters for Param objects used/updated by
   * these worekrs. Shared Param objects are associated with the same
   * ParamCounter.
   */
  typedef std::map<int, shared_ptr<ParamCounter>> ParamShard;


	void Setup(int group_id, int worker_id, shared_ptr<ParamShard> shard);

  void set_id(int group_id, int worker_id){
    group_id_=group_id;
    worker_id_=worker_id;
  }

  /**
   * @return server id where the parameter is maintained.
   */
  virtual int Sharding(int param_id);

	/**
	 * Generate a request message to Get the parameter object.
	 */
	virtual Msg* Get(shared_ptr<Param> param, int step);
  virtual Msg* Get(Msg** msg);

	/**
	 * Generate a request message to Update the parameter object.
	 */
	virtual Msg* Update(shared_ptr<Param> param, int step);
  virtual Msg* Update(Msg** msg);

	/**
	 * Collect a Param object returned from server.
	 */
	virtual Msg* Collect(Msg**);

	/**
	 * Generate a request message to Put the parameter object.
	 */
	virtual Msg* Put(shared_ptr<Param> param, int step);
  virtual Msg* Put(Msg** msg);

 protected:
  int group_id_, worker_id_;
  shared_ptr<ParamShard> shard_;
};

/**
 * Testing worker functionality.The main thread reads the config file and set up the socket.
 *
 * Create the shared ParamShard, then starts worker thread which basically carries out the work.
 * Each thread creates a PMClient object.
 *
 * The main thread then enter the loops to forward messages.
 *
 * Requests from the worker thread is prepend the paramId, which is stripped by the main thread
 * before forwarding to the correct server.
 *
 * The 1st thread in Client 0 populates the servers with data (PUT request). Wait
 * for a while before starting the client thread (which does get/update
 * continuously).
class SingaClient {
public:
	SingaClient(int worker_id, Topology &topology, vector<string> &hosts);
	void StartClient();

	int id() {
		return id_;
	}
	ParamShard *param_shard() {
		return param_shard_;
	}
	char *backend_endpoint() {
		return backend_endpoint_;
	}

private:
	int id_, local_id_, group_id_;
	char backend_endpoint_[256];
	vector<char*> neighbors_;
	ParamShard *param_shard_;

	int param_to_server_id(int paramId);//< mapping paramId to server ID
};

//Zthread function for the worker thread, in the global namespace.
//Basically a loop of: compute, get, update, compute, etc.
void ClientThread(void *args, zctx_t *ctx, void *pipe);

vector<Param*> gen_random_params();
void test_get(PMClient *client);
void test_update(PMClient *client, vector<Param*> params);
void test_collect(PMClient *client);
 */

} // namespace singa
#endif // INCLUDE_TRAINER_PM_WORKER_H_
