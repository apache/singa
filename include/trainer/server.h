#ifndef INCLUDE_TRAINER_SERVER_H_
#define INCLUDE_TRAINER_SERVER_H_
#include <memory>
#include "trainer/pm_server.h"
#include "communication/socket.h"

using std::shared_ptr;
namespace singa {
class Server{
 public:
  Server(int group_id, int server_id);
  void Setup(const UpdaterProto& proto, shared_ptr<PMServer::ParamShard> shard,
    shared_ptr<Dealer> dealer);
  void Run();

 protected:
  int group_id_, server_id_;
  shared_ptr<PMServer> pmserver_;
  shared_ptr<Dealer> dealer_;
};
} /* Server */
#endif //INCLUDE_TRAINER_SERVER_H_
