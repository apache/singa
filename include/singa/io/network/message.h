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

#ifndef SINGA_COMM_MESSAGE_H_
#define SINGA_COMM_MESSAGE_H_

#include <mutex>
#include <queue>

namespace singa {

#define MSG_DATA 0
#define MSG_ACK 1

class NetworkThread;
class EndPoint;
class Message{
    private:
        uint8_t type_;
        uint32_t id_;
        std::size_t msize_ = 0;
        std::size_t psize_ = 0;
        std::size_t processed_ = 0;
        char* msg_ = nullptr;
        static const int hsize_ = sizeof(id_) + 2 * sizeof(std::size_t) + sizeof(type_);
        char mdata_[hsize_];
        friend class NetworkThread;
        friend class EndPoint;
    public:
        Message(int = MSG_DATA, uint32_t = 0);
        Message(const Message&) = delete;
        Message(Message&&);
        ~Message();

        void setMetadata(const void*, int);
        void setPayload(const void*, int);

        std::size_t getMetadata(void**);
        std::size_t getPayload(void**);

        std::size_t getSize();
        void setId(uint32_t);
};

class MessageQueue 
{
    public:
        void push(Message&);
        Message& front();
        void pop();
        std::size_t size();
    private:
        std::mutex lock_;
        std::queue<Message> mqueue_;
};
}
#endif
