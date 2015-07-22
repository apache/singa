#ifndef INCLUDE_TRAINER_SERVER_H_
#define INCLUDE_TRAINER_SERVER_H_
#include <memory>
#include <unordered_map>
#include <utils/param.h>
#include <utils/updater.h>
#include "proto/job.pb.h"
#include "communication/socket.h"

namespace singa {
/* Repsond to worker's get/put/udpate request, and periodically syncing with
  * other servers.
  *
  * Normally, the Server creates a response message for each request which
  * will be sent back to the one who issued the request. However, if the request
  * are not processed successfully, the original message will be returned. The
  * sever does not know the returned message (response or the original message),
  * it just sends it to the router. The router will decide to re-send the
  * request to the server or send it to the worker.
  */
class Server{
 public:
  Server(int thread_id, int group_id, int server_id);
  virtual ~Server();
  void Setup(const UpdaterProto& proto,
      std::unordered_map<int, ParamEntry*>* shard,
      const vector<int>& slice2group);
  void Run();
  const int grp_id() const {
    return grp_id_;
  }
  const int id() const {
    return id_;
  }

 protected:

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
  const vector<Msg*> HandleUpdate(Msg **msg);

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
   * Generate sync message which sends local mastered Param slice to other
   * server groups
   * @param param slice to be sync with others
   * @return sync messages
   */
  const vector<Msg*> GenSyncMsgs(Param* param);

 protected:
  int thread_id_,grp_id_, id_;
  Updater* updater_;
  std::unordered_map<int, ParamEntry*> *shard_;
  vector<int> slice2group_;
  std::unordered_map<int, shared_ptr<Blob<float>>> last_data_;
  std::unordered_map<int, vector<Msg*>> buffer_requests_;
};
} /* Server */
#endif //INCLUDE_TRAINER_SERVER_H_
