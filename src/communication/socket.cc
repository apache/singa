#include "communication/socket.h"

#include <glog/logging.h>

namespace singa {

#ifdef USE_ZMQ
Poller::Poller() {
  poller_ = zpoller_new(nullptr);
}

Poller::Poller(SocketInterface* socket) {
  poller_ = zpoller_new(nullptr);
  Add(socket);
}

void Poller::Add(SocketInterface* socket) {
  zsock_t* zsock = static_cast<zsock_t*>(socket->InternalID());
  zpoller_add(poller_, zsock);
  zsock2Socket_[zsock] = socket;
}

SocketInterface* Poller::Wait(int timeout) {
  zsock_t* sock = static_cast<zsock_t*>(zpoller_wait(poller_, timeout));
  if (sock != nullptr)
    return zsock2Socket_[sock];
  else
  return nullptr;
}

bool Poller::Terminated() {
  return zpoller_terminated(poller_);
}


Dealer::Dealer() : Dealer(-1) {}

Dealer::Dealer(int id) : id_(id) {
  dealer_ = zsock_new(ZMQ_DEALER);
  CHECK_NOTNULL(dealer_);
}

Dealer::~Dealer() {
  zsock_destroy(&dealer_);
}

int Dealer::Connect(const std::string& endpoint) {
  CHECK_GT(endpoint.length(), 0);
  if (endpoint.length()) {
    CHECK_EQ(zsock_connect(dealer_, "%s", endpoint.c_str()), 0);
    return 1;
  }
  return 0;
}

int Dealer::Send(Msg** msg) {
  zmsg_t* zmsg = (*msg)->DumpToZmsg();
  zmsg_send(&zmsg, dealer_);
  delete *msg;
  *msg = nullptr;
  return 1;
}

Msg* Dealer::Receive() {
  zmsg_t* zmsg = zmsg_recv(dealer_);
  if (zmsg == nullptr)
    return nullptr;
  Msg* msg = new Msg();
  msg->ParseFromZmsg(zmsg);
  return msg;
}

void* Dealer::InternalID() const {
  return dealer_;
}

Router::Router() : Router(100) {}

Router::Router(int bufsize) {
  nBufmsg_ = 0;
  bufsize_ = bufsize;
  router_ = zsock_new(ZMQ_ROUTER);
  CHECK_NOTNULL(router_);
  poller_ = zpoller_new(router_);
  CHECK_NOTNULL(poller_);
}

Router::~Router() {
  zsock_destroy(&router_);
  for (auto it : id2addr_)
    zframe_destroy(&it.second);
  for (auto it : bufmsg_) {
    for (auto *msg : it.second)
      zmsg_destroy(&msg);
  }
}
int Router::Bind(const std::string& endpoint) {
  int port = -1;
  if (endpoint.length()) {
    port = zsock_bind(router_, "%s", endpoint.c_str());
  }
  CHECK_NE(port, -1) << endpoint;
  LOG(INFO) << "bind successfully to " << endpoint + ":" + std::to_string(port);
  return port;
}

int Router::Send(Msg **msg) {
  zmsg_t* zmsg = (*msg)->DumpToZmsg();
  int dstid = (*msg)->dst();
  if (id2addr_.find(dstid) != id2addr_.end()) {
    // the connection has already been set up
    zframe_t* addr = zframe_dup(id2addr_[dstid]);
    zmsg_prepend(zmsg, &addr);
    zmsg_send(&zmsg, router_);
  } else {
    // the connection is not ready, buffer the message
    if (bufmsg_.size() == 0)
      nBufmsg_ = 0;
    bufmsg_[dstid].push_back(zmsg);
    ++nBufmsg_;
    CHECK_LE(nBufmsg_, bufsize_);
  }
  delete *msg;
  *msg = nullptr;
  return 1;
}

Msg* Router::Receive() {
  zmsg_t* zmsg = zmsg_recv(router_);
  if (zmsg == nullptr) {
    LOG(ERROR) << "Connection broken!";
    exit(0);
  }
  zframe_t* dealer = zmsg_pop(zmsg);
  Msg* msg = new Msg();
  msg->ParseFromZmsg(zmsg);
  if (id2addr_.find(msg->src()) == id2addr_.end()) {
    // new connection, store the sender's identfier and send buffered messages
    // for it
    id2addr_[msg->src()] = dealer;
    if (bufmsg_.find(msg->src()) != bufmsg_.end()) {
      for (auto& it : bufmsg_.at(msg->src())) {
        zframe_t* addr = zframe_dup(dealer);
        zmsg_prepend(it, &addr);
        zmsg_send(&it, router_);
      }
      bufmsg_.erase(msg->src());
    }
  } else {
    zframe_destroy(&dealer);
  }
  return msg;
}

void* Router::InternalID() const {
  return router_;
}
#endif

}  // namespace singa
