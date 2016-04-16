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
#include "singa/comm/socket.h"

#include <glog/logging.h>

namespace singa {
const int TIME_OUT = 2;  // max blocking time in milliseconds.
std::unordered_map<int, SafeQueue<Msg*>> msgQueues;
Dealer::~Dealer() {
#ifdef USE_ZMQ
  zsock_destroy(&dealer_);
#endif
}

Dealer::Dealer(int id) : id_ (id) {
  msgQueues[id];
}

int Dealer::Connect(const std::string& endpoint) {
  if (endpoint.length() > 0) {
#ifdef USE_ZMQ
    dealer_ = zsock_new(ZMQ_DEALER);
    CHECK_NOTNULL(dealer_);
    CHECK_EQ(zsock_connect(dealer_, "%s", endpoint.c_str()), 0);
#else
    LOG(FATAL) << "No message passing lib is linked";
#endif
    endpoint_ = endpoint;
  }
  return 1;
}

int Dealer::Send(Msg** msg) {
  if (endpoint_.length()) {
#ifdef USE_ZMQ
    zmsg_t* zmsg = (*msg)->DumpToZmsg();
    zmsg_send(&zmsg, dealer_);
#else
    LOG(FATAL) << "No message passing lib is linked";
#endif
    delete *msg;
    *msg = nullptr;
  } else {
    msgQueues.at(-1).Push(*msg);
  }
  return 1;
}

Msg* Dealer::Receive(int timeout) {
  Msg* msg = nullptr;
  if (timeout > 0) {
    if (!msgQueues.at(id_).Pop(msg, timeout))
      return nullptr;
  } else {
    msgQueues.at(id_).Pop(msg);
  }
  msg->FirstFrame();
  return msg;
}

Router::~Router() {
#ifdef USE_ZMQ
  zsock_destroy(&router_);
#endif
}

Router::Router() {
  msgQueues[-1];
}

int Router::Bind(const std::string& endpoint) {
  int port = -1;
  if (endpoint.length() > 0) {
    endpoint_ = endpoint;
#ifdef USE_ZMQ
    router_ = zsock_new(ZMQ_ROUTER);
    CHECK_NOTNULL(router_);
    port = zsock_bind(router_, "%s", endpoint.c_str());
    CHECK_NE(port, -1) << endpoint;
    LOG(INFO) << "bind successfully to " << zsock_endpoint(router_);
    poller_ = zpoller_new(router_);
#else
    LOG(FATAL) << "No message passing lib is linked";
#endif
  }
  return port;
}

int Router::Send(Msg **msg) {
  int dstid = (*msg)->dst();
  if (msgQueues.find(dstid) != msgQueues.end()) {
    msgQueues.at(dstid).Push(*msg);
  } else {
    LOG(FATAL) << "The dst queue not exist for dstid = " << dstid;
  }
  return 1;
}

Msg* Router::Receive(int timeout) {
  Msg* msg = nullptr;
  if (timeout == 0)
    timeout = TIME_OUT;
  while (msg == nullptr) {
#ifdef USE_ZMQ
    if (router_ != nullptr) {
      zsock_t* sock = static_cast<zsock_t*>(zpoller_wait(poller_, timeout));
      if (sock != NULL) {
        zmsg_t* zmsg = zmsg_recv(router_);
        if (zmsg == nullptr) {
          LOG(ERROR) << "Connection broken!";
          exit(0);
        }
        zframe_t* dealer = zmsg_pop(zmsg);
        zframe_destroy(&dealer);
        Msg* remote_msg = new Msg();
        remote_msg->ParseFromZmsg(zmsg);
        msgQueues.at(-1).Push(remote_msg);
      }
    }
#endif
    msgQueues.at(-1).Pop(msg, timeout * 10);
  }
  msg->FirstFrame();
  return msg;
}

}  // namespace singa
