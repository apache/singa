#ifndef INCLUDE_TRAINER_SERVER_H_
#define INCLUDE_TRAINER_SERVER_H_
#include <memory>
#include <unordered_map>
#include <utils/param.h>
#include <utils/updater.h>
#include "proto/model.pb.h"
#include "communication/socket.h"

using std::shared_ptr;
namespace singa {
typedef std::unordered_map<int, shared_ptr<Param>> ServerShard;
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
  void Setup(const UpdaterProto& proto, shared_ptr<ServerShard> shard,
      const vector<int>& slice2group);
  void Run();
  const int group_id() const {
    return group_id_;
  }
  const int server_id() const {
    return server_id_;
  }

 protected:

 	/**
	 * Process GET request.
   *
   * @return the orignal message or response message
   */
	virtual Msg* HandleGet(shared_ptr<Param> param, Msg** msg);

	/**
	 * Process Update request.
   *
   * @return the orignal message or response message
   */
	virtual Msg* HandleUpdate(shared_ptr<Param> param, Msg** msg);

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
	virtual Msg* HandleSyncRequest(shared_ptr<Param> param, Msg** msg);

 protected:
  int thread_id_,group_id_, server_id_;
  shared_ptr<Dealer> dealer_;
  shared_ptr<Updater> updater_;
  shared_ptr<ServerShard> shard_;
  vector<int> slice2group_;
  std::map<int, shared_ptr<Blob<float>>> last_data_;
};
} /* Server */
#endif //INCLUDE_TRAINER_SERVER_H_
