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
#include "singa/singa_config.h"
#ifdef ENABLE_DIST

#include <arpa/inet.h>
#include <fcntl.h>
#include <netdb.h>
#include <string.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>

#include "singa/io/network.h"
#include "singa/utils/integer.h"
#include "singa/utils/logging.h"

namespace singa {

static void async_ep_cb(struct ev_loop *loop, ev_async *ev, int revent) {
  reinterpret_cast<NetworkThread *>(ev_userdata(loop))->onNewEp();
}

static void async_msg_cb(struct ev_loop *loop, ev_async *ev, int revent) {
  reinterpret_cast<NetworkThread *>(ev_userdata(loop))->onSend();
}

static void writable_cb(struct ev_loop *loop, ev_io *ev, int revent) {
  reinterpret_cast<NetworkThread *>(ev_userdata(loop))->onSend(ev->fd);
}

static void readable_cb(struct ev_loop *loop, ev_io *ev, int revent) {
  reinterpret_cast<NetworkThread *>(ev_userdata(loop))->onRecv(ev->fd);
}

static void conn_cb(struct ev_loop *loop, ev_io *ev, int revent) {
  reinterpret_cast<NetworkThread *>(ev_userdata(loop))->onConnEst(ev->fd);
}

static void accept_cb(struct ev_loop *loop, ev_io *ev, int revent) {
  reinterpret_cast<NetworkThread *>(ev_userdata(loop))->onNewConn();
}

static void timeout_cb(struct ev_loop *loop, ev_timer *ev, int revent) {
  reinterpret_cast<NetworkThread *>(ev_userdata(loop))->onTimeout(ev);
}

EndPoint::EndPoint(NetworkThread *t) : thread_(t) {
  this->timer_.data = reinterpret_cast<void *>(this);
}

EndPoint::~EndPoint() {
  while (!recv_.empty()) {
    delete send_.front();
    send_.pop();
  }
  while (!to_ack_.empty()) {
    delete send_.front();
    send_.pop();
  }
  while (!send_.empty()) {
    delete send_.front();
    send_.pop();
  }
}

int EndPoint::send(Message *msg) {
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

  send_.push(new Message(static_cast<Message &&>(*msg)));

  thread_->notify(SIG_MSG);
  return msg->getSize();
}

Message *EndPoint::recv() {
  std::unique_lock<std::mutex> lock(this->mtx_);
  while (this->recv_.empty() && conn_status_ != CONN_ERROR)
    this->cv_.wait(lock);

  Message *ret = nullptr;
  if (!recv_.empty()) {
    ret = recv_.front();
    recv_.pop();
  }
  return ret;
}

EndPointFactory::~EndPointFactory() {
  for (auto &p : ip_ep_map_) {
    delete p.second;
  }
}

EndPoint *EndPointFactory::getOrCreateEp(uint32_t ip) {
  std::unique_lock<std::mutex> lock(map_mtx_);
  if (0 == ip_ep_map_.count(ip)) {
    ip_ep_map_[ip] = new EndPoint(this->thread_);
  }
  return ip_ep_map_[ip];
}

EndPoint *EndPointFactory::getEp(uint32_t ip) {
  std::unique_lock<std::mutex> lock(map_mtx_);
  if (0 == ip_ep_map_.count(ip)) {
    return nullptr;
  }
  return ip_ep_map_[ip];
}

EndPoint *EndPointFactory::getEp(const char *host) {
  // get the ip address of host
  struct hostent *he;
  struct in_addr **list;

  if ((he = gethostbyname(host)) == nullptr) {
    LOG(INFO) << "Unable to resolve host " << host;
    return nullptr;
  }

  list = (struct in_addr **)he->h_addr_list;
  uint32_t ip = ntohl(list[0]->s_addr);

  EndPoint *ep = nullptr;
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

void EndPointFactory::getNewEps(std::vector<EndPoint *> &neps) {
  std::unique_lock<std::mutex> lock(this->map_mtx_);
  for (auto &p : this->ip_ep_map_) {
    EndPoint *ep = p.second;
    std::unique_lock<std::mutex> eplock(ep->mtx_);
    if (ep->conn_status_ == CONN_INIT) {
      neps.push_back(ep);
    }
  }
}

NetworkThread::NetworkThread(int port) {
  this->port_ = port;
  thread_ = new std::thread([this] { doWork(); });
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

  if (bind(socket_fd_, (struct sockaddr *)&addr, sizeof(addr))) {
    LOG(FATAL) << "Bind Error: " << strerror(errno);
  }

  // TODO(wangwei) remove the hardcode setting, which would result erros if
  // there are more than 10 connections
  // reported by yaochang
  if (listen(socket_fd_, 10)) {
    LOG(FATAL) << "Listen Error: " << strerror(errno);
  }

  ev_io_init(&socket_watcher_, accept_cb, socket_fd_, EV_READ);
  ev_io_start(loop_, &socket_watcher_);

  ev_set_userdata(loop_, this);

  while (1) ev_run(loop_, 0);
}

void NetworkThread::notify(int signal) {
  switch (signal) {
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
  std::vector<EndPoint *> neps;
  this->epf_->getNewEps(neps);

  for (auto &ep : neps) {
    std::unique_lock<std::mutex> ep_lock(ep->mtx_);
    int &fd = ep->fd_[0];
    if (ep->conn_status_ == CONN_INIT) {
      fd = socket(AF_INET, SOCK_STREAM, 0);
      if (fd < 0) {
        // resources not available
        LOG(FATAL) << "Unable to create socket";
      }

      // set this fd non-blocking
      fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK);

      this->fd_ep_map_[fd] = ep;

      // initialize the addess
      ep->addr_.sin_family = AF_INET;
      ep->addr_.sin_port = htons(port_);
      bzero(&(ep->addr_.sin_zero), 8);

      LOG(INFO) << "Connecting to " << inet_ntoa(ep->addr_.sin_addr)
                << " fd = " << fd;
      if (connect(fd, (struct sockaddr *)&ep->addr_, sizeof(struct sockaddr))) {
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
        afterConnEst(ep, fd, true);

        // connection established immediately
        // LOG(INFO) << "Connected to " << inet_ntoa(ep->addr_.sin_addr) << " fd
        // = "<< fd;
        // ep->conn_status_ = CONN_EST;

        // //ev_io_stop(this->loop_, &this->fd_wwatcher_map_[fd]);
        // ev_io_init(&fd_wwatcher_map_[fd], writable_cb, fd, EV_WRITE);

        // // poll for new msgs
        // ev_io_init(&this->fd_rwatcher_map_[fd], readable_cb, fd, EV_READ);
        // ev_io_start(this->loop_, &this->fd_rwatcher_map_[fd]);

        // asyncSendPendingMsg(ep);
        // ep->cv_.notify_all();
      }
    }
  }
}

void NetworkThread::onConnEst(int fd) {
  // EndPoint* ep = epf_->getEp(this->fd_ip_map_[fd]);
  CHECK(fd_ep_map_.count(fd) > 0);
  EndPoint *ep = fd_ep_map_.at(fd);

  std::unique_lock<std::mutex> lock(ep->mtx_);

  if (connect(fd, (struct sockaddr *)&ep->addr_, sizeof(struct sockaddr)) < 0 &&
      errno != EISCONN) {
    LOG(INFO) << "Unable to connect to " << inet_ntoa(ep->addr_.sin_addr)
              << ": " << strerror(errno);
    if (errno == EINPROGRESS) {
      // continue to watch this socket
      return;
    }

    handleConnLost(ep->fd_[0], ep);

    if (ep->conn_status_ == CONN_EST && ep->conn_status_ == CONN_ERROR)
      ep->cv_.notify_all();

  } else {
    afterConnEst(ep, fd, true);

    // ep->conn_status_ = CONN_EST;
    //// connect established; poll for new msgs
    // ev_io_stop(this->loop_, &this->fd_wwatcher_map_[fd]);
    // ev_io_init(&fd_wwatcher_map_[fd], writable_cb, fd, EV_WRITE);

    // ev_io_init(&this->fd_rwatcher_map_[fd], readable_cb, fd, EV_READ);
    // ev_io_start(this->loop_, &this->fd_rwatcher_map_[fd]);
  }
}

void NetworkThread::onNewConn() {
  // accept new tcp connection
  struct sockaddr_in addr;
  socklen_t len = sizeof(addr);
  int fd = accept(socket_fd_, (struct sockaddr *)&addr, &len);
  if (fd < 0) {
    LOG(INFO) << "Accept Error: " << strerror(errno);
    return;
  }

  LOG(INFO) << "Accept a client from " << inet_ntoa(addr.sin_addr)
            << ", fd = " << fd;

  // set this fd as non-blocking
  fcntl(fd, F_SETFL, fcntl(fd, F_GETFL, 0) | O_NONBLOCK);

  EndPoint *ep;
  uint32_t a = ntohl(addr.sin_addr.s_addr);

  ep = epf_->getOrCreateEp(a);
  std::unique_lock<std::mutex> lock(ep->mtx_);

  // Passive connection
  afterConnEst(ep, fd, false);

  // record the remote address
  bcopy(&addr, &ep->addr_, len);
}

void NetworkThread::onTimeout(struct ev_timer *timer) {
  EndPoint *ep = reinterpret_cast<EndPoint *>(timer->data);

  ev_tstamp timeout = EP_TIMEOUT + ep->last_msg_time_;
  ev_tstamp now = ev_now(loop_);

  std::unique_lock<std::mutex> lock(ep->mtx_);
  if (now > timeout) {
    if (!ep->to_ack_.empty() || !ep->send_.empty()) {
      LOG(INFO) << "EndPoint " << inet_ntoa(ep->addr_.sin_addr) << " timeouts";
      // we consider this ep has been disconnected
      for (int i = 0; i < 2; ++i) {
        int fd = ep->fd_[i];
        if (fd >= 0) handleConnLost(fd, ep);
      }
      return;
    }

    timer->repeat = EP_TIMEOUT;

  } else {
    timer->repeat = timeout - now;
  }

  ev_timer_again(loop_, &ep->timer_);
}

/**
 * @brief The processing for a connected socket
 *
 * @param ep
 * @param fd
 * @param active indicate whethen this socket is locally initiated or not
 */
void NetworkThread::afterConnEst(EndPoint *ep, int fd, bool active) {
  if (active)
    LOG(INFO) << "Connected to " << inet_ntoa(ep->addr_.sin_addr)
              << ", fd = " << fd;

  int sfd;

  if (active) {
    ep->fd_[0] = fd;
    sfd = ep->fd_[1];
  } else {
    if (ep->fd_[1] >= 0) {
      // the previous connection is lost
      handleConnLost(ep->fd_[1], ep, false);
    }
    ep->fd_[1] = fd;
    sfd = ep->fd_[0];
  }

  if (sfd == fd) {
    // this fd is a reuse of a previous socket fd
    // so we first need to clean the resouce for that fd
    // we duplicate this fd to let the resouce of the oldf fd can be freed
    // also indicate there is no need to reconnect
    fd = dup(fd);
    handleConnLost(sfd, ep, false);
  }

  // initialize io watchers and add the read watcher to the ev loop
  ev_io_init(&fd_rwatcher_map_[fd], readable_cb, fd, EV_READ);
  ev_io_start(loop_, &fd_rwatcher_map_[fd]);

  // stop watching the writable watcher if necessary
  if (active) ev_io_stop(loop_, &fd_wwatcher_map_[fd]);
  ev_io_init(&fd_wwatcher_map_[fd], writable_cb, fd, EV_WRITE);

  ep->last_msg_time_ = ev_now(loop_);

  // see whether there is already a established connection for this fd
  if (ep->conn_status_ == CONN_EST && sfd >= 0) {
    // check if fd and sfd are associate with the same socket
    struct sockaddr_in addr;
    socklen_t len;
    if (getsockname(fd, (struct sockaddr *)&addr, &len)) {
      LOG(INFO) << "Unable to get local socket address: " << strerror(errno);
    } else {
      // see whether the local address of fd is the same as the remote side
      // of sfd, which has already been stored in ep->addr_
      if (addr.sin_addr.s_addr == ep->addr_.sin_addr.s_addr &&
          addr.sin_port == ep->addr_.sin_port) {
        LOG(INFO) << fd << " and " << sfd
                  << " are associated with the same socket";
        ep->is_socket_loop_ = true;
      } else {
        // this socket is redundant, we close it maunally if the local ip
        // is smaller than the peer ip
        if ((addr.sin_addr.s_addr < ep->addr_.sin_addr.s_addr) ||
            (addr.sin_addr.s_addr == ep->addr_.sin_addr.s_addr &&
             addr.sin_port < ep->addr_.sin_port))
          handleConnLost(fd, ep, false);
      }
    }
  } else {
    ep->pfd_ = fd;  // set the primary fd
    ep->conn_status_ = CONN_EST;

    // start timeout watcher to detect the liveness of EndPoint
    ev_init(&ep->timer_, timeout_cb);
    ep->timer_.repeat = EP_TIMEOUT;
    ev_timer_start(loop_, &ep->timer_);
    // timeout_cb(loop_, &ep->timer_, EV_TIMER);
  }

  if (fd == ep->pfd_) {
    this->asyncSendPendingMsg(ep);
  }

  fd_ep_map_[fd] = ep;

  // Finally notify all waiting threads
  // if this connection is initiaed by remote side,
  // we dont need to notify the waiting thread
  // later threads wanting to send to this ep, however,
  // are able to reuse this ep
  if (active) {
    ep->cv_.notify_all();
  }
}

void NetworkThread::onSend(int fd) {
  std::vector<int> invalid_fd;

  if (fd == -1) {
    // LOG(INFO) << "There are " << fd_ip_map_.size() << " connections";
    // this is a signal of new message to send
    for (auto &p : fd_ep_map_) {
      // send message
      // LOG(INFO) << "Try to send over fd " << p.first;
      if (asyncSend(p.first) < 0) invalid_fd.push_back(p.first);
    }
  } else {
    if (asyncSend(fd) < 0) invalid_fd.push_back(fd);
  }

  for (auto &p : invalid_fd) {
    // EndPoint* ep = epf_->getEp(fd_ip_map_.at(p));
    EndPoint *ep = fd_ep_map_.at(p);
    std::unique_lock<std::mutex> lock(ep->mtx_);
    handleConnLost(p, ep);
  }
}

void NetworkThread::asyncSendPendingMsg(EndPoint *ep) {
  // simply put the pending msgs to the send queue

  LOG(INFO) << "There are " << ep->send_.size() << " to-send msgs, and "
            << ep->to_ack_.size() << " to-ack msgs";

  if (!ep->to_ack_.empty()) {
    while (!ep->send_.empty()) {
      ep->to_ack_.push(ep->send_.front());
      ep->send_.pop();
    }
    std::swap(ep->send_, ep->to_ack_);
  }

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
  // EndPoint* ep = epf_->getEp(fd_ip_map_[fd]);
  CHECK(fd_ep_map_.count(fd) > 0);
  EndPoint *ep = fd_ep_map_.at(fd);

  std::unique_lock<std::mutex> ep_lock(ep->mtx_);

  if (fd != ep->pfd_)
    // we only send over the primary fd
    // return -1 to indicate this fd is redundant
    return ep->is_socket_loop_ ? 0 : -1;

  if (ep->conn_status_ != CONN_EST)
    // This happens during reconnection
    goto out;

  while (!ep->send_.empty()) {
    Message &msg = *ep->send_.front();
    int nbytes;

    while (msg.processed_ < msg.getSize()) {
      if (msg.type_ == MSG_ACK) {
        nbytes = write(fd, msg.mdata_ + msg.processed_,
                       msg.getSize() - msg.processed_);
      } else
        nbytes = write(fd, msg.msg_ + msg.processed_,
                       msg.getSize() - msg.processed_);

      if (nbytes == -1) {
        if (errno == EWOULDBLOCK) {
          if (!ev_is_active(&fd_wwatcher_map_[fd]) &&
              !ev_is_pending(&fd_wwatcher_map_[fd]))
            ev_io_start(loop_, &fd_wwatcher_map_[fd]);
          goto out;
        } else {
          // this connection is lost; reset the send status
          // so that next time the whole msg would be sent entirely
          msg.processed_ = 0;
          goto err;
        }
      } else {
        ep->last_msg_time_ = ev_now(loop_);
        msg.processed_ += nbytes;
      }

      // std::size_t m, p;
      // uint8_t type;
      // uint32_t id;
      // if (msg.msg_) {
      //    readInteger(msg.msg_, type, id, m, p);
      //    LOG(INFO) << "Send " << msg.processed_ << " bytes to " <<
      // inet_ntoa(ep->addr_.sin_addr) << " over fd " << fd << " for the current
      // DATA MSG " << msg.id_ << ", " << id << ", " << m << ", " << p;
      //}
    }

    CHECK(msg.processed_ == msg.getSize());

    if (msg.type_ != MSG_ACK) {
      LOG(INFO) << "Send a DATA message to " << inet_ntoa(ep->addr_.sin_addr)
                << " for MSG " << msg.id_ << ", len = " << msg.getSize()
                << " over fd " << fd;
      msg.processed_ = 0;
      ep->to_ack_.push(&msg);
    } else {
      // LOG(INFO) << "Send an ACK message to " << inet_ntoa(ep->addr_.sin_addr)
      // << " for MSG " << msg.id_;
      delete &msg;
    }

    ep->send_.pop();

    // for test
    // if (ep->retry_cnt_ == 0) {
    //     LOG(INFO) << "Disconnect with Endpoint " <<
    // inet_ntoa(ep->addr_.sin_addr) << " over fd " << fd;
    //     close(fd);
    //     goto err;
    // }
  }
out:
  if (ep->send_.empty()) ev_io_stop(loop_, &this->fd_wwatcher_map_[fd]);
  return 0;
err:
  return -1;
}

void NetworkThread::onRecv(int fd) {
  Message *m = &pending_msgs_[fd];
  Message &msg = (*m);
  int nread;
  // EndPoint* ep = epf_->getEp(fd_ip_map_[fd]);

  CHECK(fd_ep_map_.count(fd) > 0);
  EndPoint *ep = fd_ep_map_.at(fd);

  // LOG(INFO) << "Start to read from EndPoint " <<
  // inet_ntoa(ep->addr_.sin_addr) << " over fd " << fd;

  std::unique_lock<std::mutex> lock(ep->mtx_);

  ep->last_msg_time_ = ev_now(loop_);
  while (1) {
    if (msg.processed_ < Message::hsize_) {
      nread = read(fd, msg.mdata_ + msg.processed_,
                   Message::hsize_ - msg.processed_);

      if (nread <= 0) {
        if (errno != EWOULDBLOCK || nread == 0) {
          // socket error or shuts down
          if (nread < 0)
            LOG(INFO) << "Fail to receive from EndPoint "
                      << inet_ntoa(ep->addr_.sin_addr) << ": "
                      << strerror(errno);
          else
            LOG(INFO) << "Fail to receive from EndPoint "
                      << inet_ntoa(ep->addr_.sin_addr)
                      << ": Connection reset by remote side";
          handleConnLost(fd, ep);
        }
        break;
      }

      msg.processed_ += nread;
      while (msg.processed_ >= sizeof(msg.type_) + sizeof(msg.id_)) {
        readInteger(msg.mdata_, msg.type_, msg.id_);
        if (msg.type_ == MSG_ACK) {
          LOG(INFO) << "Receive an ACK message from "
                    << inet_ntoa(ep->addr_.sin_addr) << " for MSG " << msg.id_;
          while (!ep->to_ack_.empty()) {
            Message *m = ep->to_ack_.front();
            if (m->id_ <= msg.id_) {
              delete m;
              ep->to_ack_.pop();
            } else {
              break;
            }
          }

          // reset
          msg.processed_ -= sizeof(msg.type_) + sizeof(msg.id_);
          memmove(msg.mdata_, msg.mdata_ + sizeof(msg.type_) + sizeof(msg.id_),
                  msg.processed_);

        } else
          break;
      }

      if (msg.processed_ < Message::hsize_) {
        continue;
      }

      // got the whole metadata;
      readInteger(msg.mdata_, msg.type_, msg.id_, msg.msize_, msg.psize_);

      LOG(INFO) << "Receive a message: id = " << msg.id_
                << ", msize_ = " << msg.msize_ << ", psize_ = " << msg.psize_
                << " from " << inet_ntoa(ep->addr_.sin_addr) << " over fd "
                << fd;
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
          LOG(INFO) << "Fail to receive from EndPoint "
                    << inet_ntoa(ep->addr_.sin_addr) << ": " << strerror(errno);
        else
          LOG(INFO) << "Fail to receive from EndPoint "
                    << inet_ntoa(ep->addr_.sin_addr)
                    << ": Connection reset by remote side";
        handleConnLost(fd, ep);
      }
      break;
    }

    msg.processed_ += nread;

    // LOG(INFO) << "Receive a message: id = " << msg.id_ << ", msize_ = " <<
    // msg.msize_ << ", psize_ = " << msg.psize_ << ", processed_ = " <<
    // msg.processed_ << " from " << inet_ntoa(ep->addr_.sin_addr) << " over fd
    // " << fd;

    if (msg.processed_ == msg.getSize()) {
      LOG(INFO) << "Receive a " << msg.processed_ << " bytes DATA message from "
                << inet_ntoa(ep->addr_.sin_addr) << " with id " << msg.id_;
      ep->recv_.push(new Message(static_cast<Message &&>(msg)));
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
void NetworkThread::handleConnLost(int fd, EndPoint *ep, bool reconn) {
  CHECK(fd >= 0);
  LOG(INFO) << "Lost connection to EndPoint " << inet_ntoa(ep->addr_.sin_addr)
            << ", fd = " << fd;

  this->pending_msgs_.erase(fd);
  this->fd_ep_map_.erase(fd);
  ev_io_stop(loop_, &this->fd_wwatcher_map_[fd]);
  ev_io_stop(loop_, &this->fd_rwatcher_map_[fd]);
  fd_wwatcher_map_.erase(fd);
  fd_rwatcher_map_.erase(fd);
  close(fd);

  if (fd == ep->pfd_) {
    if (!ep->send_.empty()) ep->send_.front()->processed_ = 0;
  }

  int sfd = (fd == ep->fd_[0]) ? ep->fd_[1] : ep->fd_[0];
  if (fd == ep->fd_[0])
    ep->fd_[0] = -1;
  else
    ep->fd_[1] = -1;

  if (reconn) {
    // see if the other fd is alive or not
    if (sfd < 0) {
      if (ep->conn_status_ == CONN_EST) ev_timer_stop(loop_, &ep->timer_);
      if (ep->retry_cnt_ < MAX_RETRY_CNT) {
        // notify myself for retry
        ep->retry_cnt_++;
        ep->conn_status_ = CONN_INIT;
        LOG(INFO) << "Reconnect to EndPoint " << inet_ntoa(ep->addr_.sin_addr);
        this->notify(SIG_EP);
      } else {
        LOG(INFO) << "Maximum retry count achieved for EndPoint "
                  << inet_ntoa(ep->addr_.sin_addr);
        ep->conn_status_ = CONN_ERROR;

        // notify all threads that this ep is no longer connected
        ep->cv_.notify_all();
      }
    } else {
      if (!ep->is_socket_loop_) {
        // if there is another working fd, set this fd as primary and
        // send data over this fd
        ep->pfd_ = sfd;
        ep->last_msg_time_ = ev_now(loop_);
        asyncSendPendingMsg(ep);
      } else {
        handleConnLost(sfd, ep);
      }
    }
  }
}
}  // namespace singa

#endif  // ENABLE_DIST
