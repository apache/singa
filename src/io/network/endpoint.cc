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

#include "singa/io/network/endpoint.h"
#include "singa/io/network/integer.h"
#include "singa/utils/logging.h"

#include <sys/socket.h>
#include <netdb.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <arpa/inet.h>

#include <atomic>

namespace singa {

static void async_ep_cb(struct ev_loop* loop, ev_async* ev, int revent) {
    reinterpret_cast<NetworkThread*>(ev_userdata(loop))->onNewEp();
}

static void async_msg_cb(struct ev_loop* loop, ev_async* ev, int revent) {
    reinterpret_cast<NetworkThread*>(ev_userdata(loop))->onSend();
}

static void writable_cb(struct ev_loop* loop, ev_io* ev, int revent) {
    reinterpret_cast<NetworkThread*>(ev_userdata(loop))->onSend(ev->fd);
}

static void readable_cb(struct ev_loop* loop, ev_io* ev, int revent) {
    reinterpret_cast<NetworkThread*>(ev_userdata(loop))->onRecv(ev->fd);
}

static void conn_cb(struct ev_loop* loop, ev_io* ev, int revent) {
    reinterpret_cast<NetworkThread*>(ev_userdata(loop))->onConnEst(ev->fd);
}

static void accept_cb(struct ev_loop* loop, ev_io* ev, int revent) {
    reinterpret_cast<NetworkThread*>(ev_userdata(loop))->onNewConn();
}

EndPoint::~EndPoint() {
    while(!recv_.empty()) {
        delete send_.front();
        send_.pop();
    }
    while(!to_ack_.empty()) {
        delete send_.front();
        send_.pop();
    }
    while(!send_.empty()) {
        delete send_.front();
        send_.pop();
    }
}

int EndPoint::send(Message* msg) {
    CHECK(msg->type_ == MSG_DATA);
    static std::atomic<uint32_t> id(0);
    std::unique_lock<std::mutex> lock(this->mtx_);

    if (this->conn_status_ == CONN_ERROR) {
        LOG(INFO) << "EndPoint " << inet_ntoa(addr_.sin_addr) << " is disconnected";
        return -1;
    }

    if (msg->psize_ == 0 && msg->msize_ == 0)
        // no data to send
        return 0;

    msg->setId(id++);

    send_.push(new Message(static_cast<Message&&>(*msg)));

    thread_->notify(SIG_MSG);
    return msg->getSize();
}

Message* EndPoint::recv() {
    std::unique_lock<std::mutex> lock(this->mtx_);
    while(this->recv_.empty() && conn_status_ != CONN_ERROR)
        this->cv_.wait(lock);

    Message* ret = nullptr;
    if (!recv_.empty()) {
        ret = recv_.front();
        recv_.pop();
    }
    return ret;
}

EndPointFactory::~EndPointFactory() {
    for (auto& p : ip_ep_map_) 
    {
        delete p.second;
    }
}

EndPoint* EndPointFactory::getOrCreateEp(uint32_t ip) {
    std::unique_lock<std::mutex> lock(map_mtx_);
    if (0 == ip_ep_map_.count(ip)) {
        ip_ep_map_[ip] = new EndPoint(this->thread_);
    }
    return ip_ep_map_[ip];
}

EndPoint* EndPointFactory::getEp(uint32_t ip) {
    std::unique_lock<std::mutex> lock(map_mtx_);
    if (0 == ip_ep_map_.count(ip)) {
        return nullptr;
    }
    return ip_ep_map_[ip];
}

EndPoint* EndPointFactory::getEp(const char* host) {
    // get the ip address of host
    struct hostent *he;
    struct in_addr **list;

    if ((he = gethostbyname(host)) == nullptr) {
        LOG(INFO) << "Unable to resolve host " << host;
        return nullptr;
    }

    list = (struct in_addr**) he->h_addr_list;
    uint32_t ip = ntohl(list[0]->s_addr);

    EndPoint* ep = nullptr;
    map_mtx_.lock();
    if (0 == ip_ep_map_.count(ip)) {
        ep = new EndPoint(this->thread_);
        ep->thread_ = this->thread_;
        ip_ep_map_[ip] = ep;

        // copy the address info
        bcopy(list[0], &ep->addr_.sin_addr, sizeof(struct in_addr));

        thread_->notify(SIG_EP);
    }
    ep = ip_ep_map_[ip];
    map_mtx_.unlock(); 

    std::unique_lock<std::mutex> eplock(ep->mtx_);
    while (ep->conn_status_ == CONN_PENDING || ep->conn_status_ == CONN_INIT) {
        ep->pending_cnt_++;
        ep->cv_.wait(eplock);
        ep->pending_cnt_--;
    }

    if (ep->conn_status_ == CONN_ERROR) {
        ep = nullptr;
    }

    return ep;
}

void EndPointFactory::getNewEps(std::vector<EndPoint*>& neps) {
    std::unique_lock<std::mutex> lock(this->map_mtx_);
    for (auto& p : this->ip_ep_map_) {
        EndPoint* ep = p.second;
        std::unique_lock<std::mutex> eplock(ep->mtx_);
        if (ep->conn_status_ == CONN_INIT) {
            neps.push_back(ep);
        }
    } 
}

NetworkThread::NetworkThread(int port) {
    this->port_ = port;
    thread_ = new std::thread([this] {doWork();});
    this->epf_ = new EndPointFactory(this);
}

void NetworkThread::doWork() {

    // prepare event loop
    if (!(loop_ = ev_default_loop(0))) {
        // log here
    }

    ev_async_init(&ep_sig_, async_ep_cb);
    ev_async_start(loop_, &ep_sig_);

    ev_async_init(&msg_sig_, async_msg_cb);
    ev_async_start(loop_, &msg_sig_);

    // bind and listen
    struct sockaddr_in addr;
    if ((socket_fd_ = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        LOG(FATAL) << "Socket Error: " << strerror(errno);
    }

    bzero(&addr, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons(this->port_);
    addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(socket_fd_, (struct sockaddr*)&addr, sizeof(addr))) {
        LOG(FATAL) << "Bind Error: " << strerror(errno);
    }

    if (listen(socket_fd_, 10)) {
        LOG(FATAL) << "Listen Error: " << strerror(errno);
    }

    ev_io_init(&socket_watcher_, accept_cb, socket_fd_, EV_READ);
    ev_io_start(loop_, &socket_watcher_);

    ev_set_userdata(loop_, this);

    while(1)
        ev_run(loop_, 0);
}

void NetworkThread::notify(int signal) {
    switch(signal) {
        case SIG_EP:
            ev_async_send(this->loop_, &this->ep_sig_);
            break;
        case SIG_MSG:
            ev_async_send(this->loop_, &this->msg_sig_);
            break;
        default:
            break;
    }
}

void NetworkThread::onNewEp() {
    std::vector<EndPoint*> neps;
    this->epf_->getNewEps(neps);

    for (auto& ep : neps) {
        std::unique_lock<std::mutex>  ep_lock(ep->mtx_);
        int& fd = ep->fd_[0];
        if (ep->conn_status_ == CONN_INIT) {

            fd = socket(AF_INET, SOCK_STREAM, 0);
            if (fd < 0) {
                // resources not available
                LOG(FATAL) << "Unable to create socket";
            }

            // set this fd non-blocking
            fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK);

            this->fd_ip_map_[fd] = ntohl(ep->addr_.sin_addr.s_addr);

            // initialize the addess
            ep->addr_.sin_family = AF_INET;
            ep->addr_.sin_port = htons(port_);
            bzero(&(ep->addr_.sin_zero), 8);

            if (connect(fd, (struct sockaddr*)&ep->addr_, 
                    sizeof(struct sockaddr)) ) {
                LOG(INFO) << "Connect Error: " << strerror(errno);
                if (errno != EINPROGRESS) {
                    ep->conn_status_ = CONN_ERROR;
                    ep->cv_.notify_all();
                    continue;
                } else {
                    ep->conn_status_ = CONN_PENDING;
                    ev_io_init(&this->fd_wwatcher_map_[fd], conn_cb, fd, EV_WRITE);
                    ev_io_start(this->loop_, &this->fd_wwatcher_map_[fd]);
                }
            } else {
                // connection established immediately
                LOG(INFO) << "Connected to " << inet_ntoa(ep->addr_.sin_addr) << " fd = "<< fd;
                ep->conn_status_ = CONN_EST;
                ev_io_stop(this->loop_, &this->fd_wwatcher_map_[fd]);

                // poll for new msgs
                ev_io_init(&this->fd_rwatcher_map_[fd], readable_cb, fd, EV_READ);
                ev_io_start(this->loop_, &this->fd_rwatcher_map_[fd]);

                asyncSendPendingMsg(ep);
                ep->cv_.notify_all();
            }
        }
    }
}

void NetworkThread::onConnEst(int fd) {

    EndPoint* ep = epf_->getEp(this->fd_ip_map_[fd]);

    std::unique_lock<std::mutex> lock(ep->mtx_);

    if (connect(fd, (struct sockaddr*)&ep->addr_, sizeof(struct sockaddr)) < 0 && errno != EISCONN) {
        LOG(INFO) << "Unable to connect to " << inet_ntoa(ep->addr_.sin_addr) << ": "<< strerror(errno);
        if (errno == EINPROGRESS) {
            // continue to watch this socket
            return;
        }

        handleConnLost(ep->fd_[0], ep);

        switch(ep->conn_status_) {
            case CONN_INIT:
            case CONN_PENDING:
                return;
            default:
                break;
        }

    } else {
        LOG(INFO) << "Connected to " << inet_ntoa(ep->addr_.sin_addr) << ", fd = "<< fd;
        ep->conn_status_ = CONN_EST;
        // connect established; poll for new msgs
        ev_io_stop(this->loop_, &this->fd_wwatcher_map_[fd]);

        ev_io_init(&this->fd_rwatcher_map_[fd], readable_cb, fd, EV_READ);
        ev_io_start(this->loop_, &this->fd_rwatcher_map_[fd]);
    }

    if (ep->conn_status_ == CONN_EST && ep->to_ack_.size() > 0)
        // if there are pending message, it means these msgs were sent over
        // previous sockets that have been lost now
        // we need to resend these msgs to the remote side
        asyncSendPendingMsg(ep);

    // Finally notify all waiting threads
    ep->cv_.notify_all();
}

void NetworkThread::onNewConn() {
    // accept new tcp connection
    struct sockaddr_in addr;
    socklen_t len = sizeof(addr);
    int fd = accept(socket_fd_, (struct sockaddr*)&addr, &len);
    if (fd < 0) {
        LOG(INFO) << "Accept Error: " << strerror(errno);
        return;
    }

    LOG(INFO) << "Accept a client from " << inet_ntoa(addr.sin_addr) << ", fd = " << fd;

    // set this fd as non-blocking
    fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK);

    EndPoint* ep;
    uint32_t a = ntohl(addr.sin_addr.s_addr);

    ep = epf_->getOrCreateEp(a); 
    std::unique_lock<std::mutex> lock(ep->mtx_);

    if (ep->fd_[1] >= 0) {
        // the previous connection is lost
        handleConnLost(ep->fd_[1], ep, false);
    }

    if (ep->fd_[0] == fd) {
        // this fd is reused
        handleConnLost(fd, ep, false);
    }

    fd_ip_map_[fd] = a;
    ev_io_init(&fd_rwatcher_map_[fd], readable_cb, fd, EV_READ);
    ev_io_start(loop_, &fd_rwatcher_map_[fd]);

    // record the remote address
    bcopy(&addr, &ep->addr_, len);

    ep->conn_status_ = CONN_EST;
    ep->fd_[1] = fd;

    if (ep->to_ack_.size() > 0)
        // see if there are any messages waiting for ack
        // if yes, resend them
        asyncSendPendingMsg(ep);

    // this connection is initiaed by remote side, 
    // so we dont need to notify the waiting thread
    // later threads wanting to send to this ep, however,
    // are able to reuse this ep
}

void NetworkThread::onSend(int fd) {
    std::vector<int> invalid_fd;

    if (fd == -1) {
        // this is a signal of new message to send
        for(auto& p : fd_ip_map_) {
            // send message
            if (asyncSend(p.first) < 0)
                invalid_fd.push_back(p.first);
        }
    } else {
        if (asyncSend(fd) < 0)
            invalid_fd.push_back(fd);
    }

    for (auto& p : invalid_fd) {
        EndPoint* ep = epf_->getEp(fd_ip_map_.at(p));
        std::unique_lock<std::mutex> lock(ep->mtx_);
        handleConnLost(p, ep);
    } 
}

void NetworkThread::asyncSendPendingMsg(EndPoint* ep) {
    // simply put the pending msgs to the send queue

    LOG(INFO) << "There are " << ep->send_.size() << " to-send msgs, and " << ep->to_ack_.size() << " to-ack msgs";

    while (!ep->send_.empty()) {
        ep->to_ack_.push(ep->send_.front());
        ep->send_.pop();
    }

    std::swap(ep->send_, ep->to_ack_);

    if (ep->send_.size() > 0) {
        notify(SIG_MSG);
    }
}

/**
 * @brief non-locking send; 
 *
 * @param ep
 *
 */
int NetworkThread::asyncSend(int fd) {

    EndPoint* ep = epf_->getEp(fd_ip_map_[fd]);

    std::unique_lock<std::mutex> ep_lock(ep->mtx_);

    if (ep->conn_status_ != CONN_EST)
        goto out;

    while(!ep->send_.empty()) {

        Message& msg = *ep->send_.front();
        int nbytes;

        while(msg.processed_ < msg.getSize()) {
            if (msg.type_ == MSG_ACK) {
                nbytes = write(fd, msg.mdata_ + msg.processed_, msg.getSize() - msg.processed_);
            }
            else
                nbytes = write(fd, msg.msg_ + msg.processed_, msg.getSize() - msg.processed_);

            LOG(INFO) << "Send " << nbytes << " bytes to " << inet_ntoa(ep->addr_.sin_addr) << " over fd " << fd;

            if (nbytes == -1) {
                if (errno == EWOULDBLOCK) {
                    ev_io_init(&fd_wwatcher_map_[fd], writable_cb, fd, EV_WRITE);
                    ev_io_start(loop_, &fd_wwatcher_map_[fd]);
                    goto out;
                } else {
                    // this connection is lost; reset the send status
                    // so that next time the whole msg would be sent entirely
                    msg.processed_ = 0;
                    goto err;
                }
            } else 
                msg.processed_ += nbytes;
        }

        CHECK(msg.processed_ == msg.getSize());

        if (msg.type_ != MSG_ACK) {
            msg.processed_ = 0;
            ep->to_ack_.push(&msg);
        } else {
            delete &msg;
        }

        ep->send_.pop();

        // for test
        if (ep->retry_cnt_ == 0)
            close(fd);
    }
out:
    if (ep->send_.empty())
        ev_io_stop(loop_, &this->fd_wwatcher_map_[fd]);
    return 0;
err:
    return -1;
}

void NetworkThread::onRecv(int fd) {

    Message* m = &pending_msgs_[fd];
    Message& msg = (*m);
    int nread;
    EndPoint* ep = epf_->getEp(fd_ip_map_[fd]);

    LOG(INFO) << "Start to read from EndPoint " << inet_ntoa(ep->addr_.sin_addr) << " over fd " << fd; 

    std::unique_lock<std::mutex> lock(ep->mtx_);
    while(1) {

        if (msg.processed_ < Message::hsize_) {
            nread = read(fd, msg.mdata_ + msg.processed_, 
                    Message::hsize_ - msg.processed_);

            if (nread <= 0) {
                if (errno != EWOULDBLOCK || nread == 0) {
                    // socket error or shuts down 
                    if (nread < 0)
                        LOG(INFO) << "Faile to receive from EndPoint " << inet_ntoa(ep->addr_.sin_addr) << ": " << strerror(errno);
                    else
                        LOG(INFO) << "Faile to receive from EndPoint " << inet_ntoa(ep->addr_.sin_addr) << ": Connection reset by remote side";
                    handleConnLost(fd, ep);
                }
                break;
            }

            msg.processed_ += nread;
            while (msg.processed_ >= sizeof(msg.type_) + sizeof(msg.id_)) {
                readInteger(msg.mdata_, msg.type_, msg.id_);
                if(msg.type_ == MSG_ACK) {
                    LOG(INFO) << "Receive an ACK message from " << inet_ntoa(ep->addr_.sin_addr) << " for MSG " << msg.id_;
                    while (!ep->to_ack_.empty()) {
                        Message* m = ep->to_ack_.front();
                        if (m->id_ <= msg.id_) {
                            delete m;
                            ep->to_ack_.pop();
                        } else {
                            break;
                        }
                    }

                    // reset
                    msg.processed_ -= sizeof(msg.type_) + sizeof(msg.id_);
                    memmove(msg.mdata_, 
                            msg.mdata_ + sizeof(msg.type_) + sizeof(msg.id_),
                            msg.processed_);

                } else break;
            }

            if (msg.processed_ < Message::hsize_) {
                continue;
            }

            // got the whole metadata; 
            readInteger(msg.mdata_, msg.type_, msg.id_, msg.msize_, msg.psize_);
            LOG(INFO) << "Receive a message: id = " << msg.id_ << ", msize_ = " << msg.msize_ << ", psize_ = " << msg.psize_;
        }

        // start reading the real data
        if (msg.msg_ == nullptr) {
            msg.msg_ = new char[msg.getSize()];
            memcpy(msg.msg_, msg.mdata_, Message::hsize_);
        }

        nread = read(fd, msg.msg_ + msg.processed_, msg.getSize() - msg.processed_);
        if (nread <= 0) {
            if (errno != EWOULDBLOCK || nread == 0) {
                // socket error or shuts down 
                if (nread < 0)
                    LOG(INFO) << "Faile to receive from EndPoint " << inet_ntoa(ep->addr_.sin_addr) << ": " << strerror(errno);
                else
                    LOG(INFO) << "Faile to receive from EndPoint " << inet_ntoa(ep->addr_.sin_addr) << ": Connection reset by remote side";
                handleConnLost(fd, ep);
            }
            break;
        }

        msg.processed_ += nread;
        if (msg.processed_ == msg.getSize()) {
            LOG(INFO) << "Receive a " << msg.processed_ << " bytes DATA message from " << inet_ntoa(ep->addr_.sin_addr) << " with id " << msg.id_;
            ep->recv_.push(new Message(static_cast<Message&&>(msg)));
            // notify of waiting thread
            ep->cv_.notify_one();
            ep->send_.push(new Message(MSG_ACK, msg.id_));
            msg.processed_ = 0;
        }
    }
}

/**
 * @brief clean up for the lost connection; the caller should acquire the lock
 * for the respective endpoint
 *
 * @param fd
 * @param ep
 * @param reconn
 */
void NetworkThread::handleConnLost(int fd, EndPoint* ep, bool reconn) {
    CHECK(fd >= 0);
    LOG(INFO) << "Lost connection to EndPoint " << inet_ntoa(ep->addr_.sin_addr) << ", fd = " << fd;

    this->pending_msgs_.erase(fd);
    this->fd_ip_map_.erase(fd);
    ev_io_stop(loop_, &this->fd_wwatcher_map_[fd]);
    ev_io_stop(loop_, &this->fd_rwatcher_map_[fd]);
    fd_wwatcher_map_.erase(fd);
    fd_rwatcher_map_.erase(fd);
    close(fd);

    if (fd == ep->fd_[0] || ep->fd_[0] < 0) {
        if (!ep->send_.empty())
            ep->send_.front()->processed_ = 0;
    }

    if (reconn) {

        int sfd = (fd == ep->fd_[0]) ? ep->fd_[1] : ep->fd_[0];

        if (fd == ep->fd_[0])
            ep->fd_[0] = -1;
        else
            ep->fd_[1] = -1;

        // see if the other fd is ok or not
        if (sfd < 0) {
            if (ep->retry_cnt_ < MAX_RETRY_CNT) {
                // notify myself for retry
                ep->retry_cnt_++;
                ep->conn_status_ = CONN_INIT;
                LOG(INFO) << "Reconnect to EndPoint " << inet_ntoa(ep->addr_.sin_addr);
                this->notify(SIG_EP);
            } else {
                LOG(INFO) << "Maximum retry count achieved for EndPoint " << inet_ntoa(ep->addr_.sin_addr);
                ep->conn_status_ = CONN_ERROR;
                // notify all threads that this ep is no longer connected
                ep->cv_.notify_all();
            }
        } else {
            // if there is another working fd, try to send data over this fd
            if (!ep->send_.empty())
                this->notify(SIG_MSG);
        }
    }
}

}
