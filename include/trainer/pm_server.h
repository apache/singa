#ifndef INCLUDE_TRAINER_PM_SERVER_H_
#define INCLUDE_TRAINER_PM_SERVER_H_

#include <czmq.h>
#include <memory>
#include <vector>
#include <map>
#include <string.h>
#include "proto/model.pb.h"
#include "utils/updater.h"
#include "utils/param.h"
#include "communication/msg.h"
#include "communication/socket.h"
using std::vector;
using std::string;
using std::shared_ptr;

namespace singa{

/**
 * Parameter manager at the server side.
 *
 * Repsond to worker's get/put/udpate request, and periodically syncing with
 * other servers.
 *
 * Normally, the PMServer creates a response message for each request which
 * will be sent back to the one who issued the request. However, if the request
 * are not processed successfully, the original message will be returned. The
 * sever does not know the returned message (response or the original message),
 * it just sends it to the router. The router will decide to re-send the
 * request to the server or send it to the worker.
 *
 */
class PMServer{
public:
  typedef std::map<int, shared_ptr<Param>> ParamShard;

	void Setup(int group_id, int server_id, shared_ptr<ParamShard> shard,
       const UpdaterProto& proto);

	~PMServer();

	/**
	 * Process GET request.
   *
   * @return the orignal message or response message
   */
	virtual Msg* HandleGet(Msg** msg);

	/**
	 * Process Update request.
   *
   * @return the orignal message or response message
   */
	virtual Msg* HandleUpdate(Msg** msg);

	/**
	 * Process PUT request.
   *
   * @return the original message or response message. If we don't want need to
   * acknowledge the put request, then return nullptr.
	 */
	virtual Msg* HandlePut(Msg **msg);

	/**
   * TODO Process SYNC request.
	 */
	virtual Msg* HandleSyncRequest(Msg** msg);

	/**
   * TODO Process SYNC response.
	 */
	virtual int HandleSyncResponse(Msg** msg);

  /**
   * Scheduler for synchronizing server groups.
   *
   * TODO implement the Caffe's synchronization scheduler for data parallelism
   */
  virtual bool SyncNow();

 protected:
  int group_id_, server_id_;
  shared_ptr<ParamShard> shard_;
  shared_ptr<Dealer> dealer_;
  shared_ptr<Updater> updater_;
};

} // namespace singa

#endif // INCLUDE_TRAINER_PM_SERVER_H_
