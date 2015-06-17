#ifndef SINGA_COMMUNICATION_SOCKET_H_
#define SINGA_COMMUNICATION_SOCKET_H_

#include <map>
#include <string>
#include <vector>

#include "communication/msg.h"

#ifdef USE_ZMQ
#include <czmq.h>
#endif

namespace singa {

const std::string kInprocRouterEndpoint = "inproc://router";

class SocketInterface {
 public:
  virtual ~SocketInterface() {}
  /**
    * Send a message to connected socket(s), non-blocking. The message
    * will be deallocated after sending, thus should not be used after
    * calling Send();
    *
    * @param msg The message to be sent
    * @return 1 for success queuing the message for sending, 0 for failure
    */
  virtual int Send(Msg** msg) = 0;
  /**
    * Receive a message from any connected socket.
    *
    * @return a message pointer if success; nullptr if failure
    */
  virtual Msg* Receive() = 0;
  /**
   * @return Identifier of the implementation dependent socket. E.g., zsock_t*
   * for ZeroMQ implementation and rank for MPI implementation.
   */
  virtual void* InternalID() const = 0;
};

class Poller {
 public:
  Poller();
  /**
    * Add a socket for polling; Multiple sockets can be polled together by
    * adding them into the same poller.
    */
  void Add(SocketInterface* socket);
  /**
    * Poll for all sockets added into this poller.
    * @param timeout Stop after this number of mseconds
    * @return pointer To the socket if it has one message in the receiving
    * queue; nullptr if no message in any sockets,
    */
  SocketInterface* Wait(int duration);

  /**
   * @return true if the poller is terminated due to process interupt
   */
  virtual bool Terminated()=0;

 protected:
#ifdef USE_ZMQ
  zpoller_t *poller_;
  std::map<zsock_t*, SocketInterface*> zsock2Socket_;
#endif
};

class Dealer : public SocketInterface {
 public:
  /*
   * @param id Local dealer ID within a procs if the dealer is from worker or
   * server thread, starts from 1 (0 is used by the router); or the connected
   * remote procs ID for inter-process dealers from the stub thread.
   */
  Dealer();
  explicit Dealer(int id);
  ~Dealer() override;
  /**
    * Setup the connection with the router.
    *
    * @param endpoint Identifier of the router. For intra-process
    * connection, the endpoint follows the format of ZeroMQ, i.e.,
    * starting with "inproc://"; in Singa, since each process has one
    * router, hence we can fix the endpoint to be "inproc://router" for
    * intra-process. For inter-process, the endpoint follows ZeroMQ's
    * format, i.e., IP:port, where IP is the connected process.
    * @return 1 connection sets up successfully; 0 otherwise
    */
  int Connect(const std::string& endpoint);
  int Send(Msg** msg) override;
  Msg* Receive() override;
  void* InternalID() const override;

 protected:
  int id_ = -1;
#ifdef USE_ZMQ
  zsock_t* dealer_ = nullptr;
  zpoller_t* poller_ = nullptr;
#endif
};

class Router : public SocketInterface {
 public:
  Router();
  /**
   * There is only one router per procs, hence its local id is 0 and is not set
   * explicitly.
   *
   * @param bufsize Buffer at most this number of messages
   */
  explicit Router(int bufsize);
  ~Router() override;
  /**
   * Setup the connection with dealers.
   *
   * It automatically binds to the endpoint for intra-process communication,
   * i.e., "inproc://router".
   *
   * @param endpoint The identifier for the Dealer socket in other process
   * to connect. It has the format IP:Port, where IP is the host machine.
   * If endpoint is empty, it means that all connections are
   * intra-process connection.
   * @return number of connected dealers.
   */
  int Bind(const std::string& endpoint);
  /**
   * If the destination socket has not connected yet, buffer this the message.
   */
  int Send(Msg** msg) override;
  Msg* Receive() override;
  void* InternalID() const override;

 protected:
  int nBufmsg_ = 0;
  int bufsize_ = 100;
#ifdef USE_ZMQ
  zsock_t* router_ = nullptr;
  zpoller_t* poller_ = nullptr;
  std::map<int, zframe_t*> id2addr_;
  std::map<int, std::vector<zmsg_t*>> bufmsg_;
#endif
};

#ifdef USE_MPI
// TODO(wangsheng): add intra-process communication using shared queue
std::vector<SafeQueue*> MPIQueues;
#endif

}  // namespace singa

#endif  // SINGA_COMMUNICATION_SOCKET_H_
