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

#include <atomic>
#include <cstdlib>
#include <cstring>

#include "singa/io/network.h"
#include "singa/utils/integer.h"

namespace singa {

Message::Message(Message &&msg) {
  std::swap(msize_, msg.msize_);
  std::swap(psize_, msg.psize_);
  std::swap(msg_, msg.msg_);
  std::swap(type_, msg.type_);
  std::swap(id_, msg.id_);
}

Message::Message(int type, uint32_t ack_msg_id) : type_(type), id_(ack_msg_id) {
  if (type_ == MSG_ACK) appendInteger(mdata_, type_, id_);
}

Message::~Message() {
  if (msg_) free(msg_);
}

std::size_t Message::getSize() {
  if (type_ == MSG_ACK)
    return sizeof(type_) + sizeof(id_);
  else
    return this->hsize_ + this->psize_ + this->msize_;
}

void Message::setId(uint32_t id) {
  this->id_ = id;
  appendInteger(msg_, type_, id_);
}

void Message::setMetadata(const void *buf, int size) {
  this->msize_ = size;
  msg_ = (char *)malloc(this->getSize());
  appendInteger(msg_, type_, id_, msize_, psize_);
  memcpy(msg_ + hsize_, buf, size);
}

void Message::setPayload(const void *buf, int size) {
  this->psize_ = size;
  msg_ = (char *)realloc(msg_, this->getSize());
  appendInteger(msg_ + hsize_ - sizeof(psize_), psize_);
  memcpy(msg_ + hsize_ + msize_, buf, size);
}

std::size_t Message::getMetadata(void **p) {
  if (this->msize_ == 0)
    *p = nullptr;
  else
    *p = msg_ + hsize_;
  return this->msize_;
}

std::size_t Message::getPayload(void **p) {
  if (this->psize_ == 0)
    *p = nullptr;
  else
    *p = msg_ + hsize_ + msize_;
  return this->psize_;
}
}  // namespace singa

#endif  // ENABLE_DIST
