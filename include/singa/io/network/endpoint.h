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

#ifndef SINGA_COMM_END_POINT_H_
#define SINGA_COMM_END_POINT_H_

#include <ev.h>
#include <thread>
#include <unordered_map>
#include <map>
#include <vector>
#include <condition_variable>
#include <mutex>
#include <atomic>
#include <string>
#include <netinet/in.h>

#include "singa/io/network/message.h"

namespace singa {

#define LOCKED 1
#define UNLOCKED 0

#define SIG_EP 1
#define SIG_MSG 2

#define CONN_INIT 0
#define CONN_PENDING 1
#define CONN_EST 2
#define CONN_ERROR 3

#define MAX_RETRY_CNT 3

class NetworkThread;
class EndPointFactory;

class EndPoint {
    private:
        std::queue<Message*> send_;
        std::queue<Message*> recv_;
        std::queue<Message*> to_ack_;
        std::condition_variable cv_;
        std::mutex mtx_;
        struct sockaddr_in addr_;
        int fd_[2] = {-1, -1}; // two endpoints simultaneously connect to each other
        int conn_status_ = CONN_INIT;
        int pending_cnt_ = 0;
        int retry_cnt_ = 0;
        NetworkThread* thread_ = nullptr;
        EndPoint(NetworkThread* t):thread_(t){}
        ~EndPoint();
        friend class NetworkThread;
        friend class EndPointFactory;
    public:
        int send(Message*);
        Message* recv();
};

class EndPointFactory {
    private:
        std::unordered_map<uint32_t, EndPoint*> ip_ep_map_;
        std::condition_variable map_cv_;
        std::mutex map_mtx_;
        NetworkThread* thread_;
        EndPoint* getEp(uint32_t ip);
        EndPoint* getOrCreateEp(uint32_t ip);
        friend class NetworkThread;
    public:
        EndPointFactory(NetworkThread* thread) : thread_(thread) {}
        ~EndPointFactory();
        EndPoint* getEp(const char* host);
        void getNewEps(std::vector<EndPoint*>& neps);
};

class NetworkThread{
    private:
        struct ev_loop *loop_;
        ev_async ep_sig_;
        ev_async msg_sig_;

        std::thread* thread_;

        std::unordered_map<int, ev_io> fd_wwatcher_map_;
        std::unordered_map<int, ev_io> fd_rwatcher_map_;
        std::unordered_map<int, uint32_t> fd_ip_map_;

        std::map<int, Message> pending_msgs_;

        ev_io socket_watcher_;
        int port_;
        int socket_fd_;

        void handleConnLost(int, EndPoint*, bool = true);
        void doWork();
        int asyncSend(int);
        void asyncSendPendingMsg(EndPoint*);
    public:
        EndPointFactory* epf_;

        NetworkThread(int);
        //void join();
        void notify(int signal);

        void onRecv(int fd);
        void onSend(int fd = -1);
        void onConnEst(int fd);
        void onNewEp();
        void onNewConn();
};
}
#endif
