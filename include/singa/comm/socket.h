/************************************************************
*
* Licensed to the Apache Software Foundation (ASF) under one
* or more contributor license agreements.  See the NOTICE file
* distributed with this work for additional information
* regarding copyright ownership.  The ASF licenses this file
* to you under the Apache License, Version 2.0 (the
* "License"); you may not use this file except in compliance
* with the License.  You may obtain a copy of the License at
*
*   http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*
*************************************************************/

#ifndef SINGA_COMM_SOCKET_H_
#define SINGA_COMM_SOCKET_H_

#ifdef USE_ZMQ
#include <czmq.h>
#endif

#include <map>
#include <string>
#include <vector>
#include <unordered_map>
#include "singa/utils/safe_queue.h"
#include "singa/comm/msg.h"

namespace singa {
/**
 * Worker and Server use Dealer to communicate with Stub.
 * Stub uses Dealer to communicate with remote Stub.
 */
class Dealer {
 public:
   /**
    * @param id used for identifying the msg queue of this dealer.
    */
   explicit Dealer(int id);
  ~Dealer();
  /**
   * Setup the connection with the remote router.
   *
   * For local router, there is no need to connect it.
   *
   * @param endpoint Identifier of the remote router to connect. It follows
   * ZeroMQ's format, i.e., IP:port, where IP is the connected process.
   * @return 1 connection sets up successfully; 0 otherwise
   */
  int Connect(const std::string& endpoint);
  /**
   * Send a message to the local router (id=-1) or remote outer. It is
   * non-blocking. The message will be deallocated after sending, thus
   * should not be used after calling Send();
   */
  int Send(Msg** msg);
  /**
   * Recv msg from local router.
   *
   * @param timeout return if waiting for timeout microseconds.
   * @return a message pointer if success; nullptr if failure
   */
  Msg* Receive(int timeout = 0);

 protected:
  std::string endpoint_;
  int id_;
#ifdef USE_ZMQ
  zsock_t* dealer_ = nullptr;
#endif
};
/**
 * In Singa, since each process has one router used by Stub, hence we fix the
 * router to use the msg queue indexed by -1.
 */
class Router {
 public:
  ~Router();
  Router();
  /**
   * Bind the router to an endpoint for recv msg from remote dealer.
   * If the router is used for intra-communication only, then no need to call
   * Bind.
   *
   * @param endpoint identifier for the Dealer socket in other process
   * to connect. It has the format IP:Port, where IP is the host machine.
   * @return number of connected dealers.
   */
  int Bind(const std::string& endpoint);
  /**
   * Send msg to local dealers by pushing the msg into the msg queue indexed by
   * dst of the msg.
   */
  int Send(Msg** msg);
  /**
   * Recv msg from local (msg queue) or remote dealer (via zmq).
   */
  Msg* Receive(int timeout = 0);

 protected:
  std::string endpoint_;
#ifdef USE_ZMQ
  zsock_t* router_ = nullptr;
  zpoller_t* poller_ = nullptr;
#endif
};

/**
 * Used for intra-process communication.
 * Each dealer/router has a SafeQueue for recieving msgs.
 * The sender pushes msgs onto the queue of the reciever's queue.
 */
extern std::unordered_map<int, SafeQueue<Msg*>> msgQueues;
}  // namespace singa

#endif  // SINGA_COMM_SOCKET_H_
