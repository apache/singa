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

#ifndef SINGA_UTILS_SAFE_QUEUE_H_
#define SINGA_UTILS_SAFE_QUEUE_H_

#include <algorithm>
#include <condition_variable>
#include <list>
#include <mutex>
#include <queue>
#include <thread>

/**
 * Thread-safe queue.
 */
template <typename T, class Container = std::queue<T>>
class SafeQueue {
 public:
  SafeQueue() = default;
  ~SafeQueue() { std::lock_guard<std::mutex> lock(mutex_); }

  /**
   * Push an element into the queue. Blocking operation.
   * @return true if success;
   */
  bool Push(const T& e) {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.push(e);
    condition_.notify_one();
    return true;
  }

  /**
   * Pop an element from the queue.
   * It will be blocked until one element is poped.
   */
  void Pop(T& e) {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this]() { return !queue_.empty(); });
    e = queue_.front();
    queue_.pop();
  }
  /**
   * Pop an item from the queue until one element is poped or timout.
   * @param[in] timeout, return false after waiting this number of microseconds
   */
  bool Pop(T& item, std::uint64_t timeout) {
    std::unique_lock<std::mutex> lock(mutex_);

    if (queue_.empty()) {
      if (timeout == 0) return false;

      if (condition_.wait_for(lock, std::chrono::microseconds(timeout)) ==
          std::cv_status::timeout)
        return false;
    }

    item = queue_.front();
    queue_.pop();
    return true;
  }

  /**
   *  Try to pop an element from the queue.
   * \return false the queue is empty now.
   */
  bool TryPop(T& e) {
    std::unique_lock<std::mutex> lock(mutex_);

    if (queue_.empty()) return false;

    e = queue_.front();
    queue_.pop();
    return true;
  }

  /**
   * @return Number of elements in the queue.
   */
  unsigned int Size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
  }

 private:
  Container queue_;
  mutable std::mutex mutex_;
  std::condition_variable condition_;
};

/**
 * Thread safe priority queue.
 */
template <typename T>
class PriorityQueue {
 public:
  PriorityQueue() = default;
  /**
   * Push an element into the queue with a given priority.
   * The queue should not be a priority queue.
   * @return true if success; otherwise false, e.g., due to capacity constraint.
   */
  bool Push(const T& e, int priority) {
    Element ele;
    ele.data = e;
    ele.priority = priority;
    queue_.push(ele);
    return true;
  }

  /**
   * Pop an element from the queue with the highest priority.
   * It blocks until one element is poped.
   */
  void Pop(T& e) {
    Element ele;
    queue_.pop(ele);
    e = ele.data;
  }
  /**
   * Pop the item with the highest priority from the queue until one element is
   * poped or timeout.
   * @param[in] timeout, return false if no element is poped after this number
   * of microseconds.
   */
  bool Pop(T& e, std::uint64_t timeout) {
    Element ele;
    if (queue_.pop(ele, timeout)) {
      e = ele.data;
      return true;
    } else {
      return false;
    }
  }

  /**
   * Try to pop an element from the queue.
   * @return false if the queue is empty now.
   */
  bool TryPop(T& e) {
    Element ele;
    if (queue_.TryPop(ele)) {
      e = ele.data;
      return true;
    } else {
      return false;
    }
  }

  /**
   * @return Number of elements in the queue.
   */
  unsigned int Size() const { return queue_.Size(); }

 private:
  struct Element {
    T data;
    int priority;
    inline bool operator<(const Element& other) const {
      return priority < other.priority;
    }
  };

  SafeQueue<Element, std::priority_queue<Element>> queue_;
};

#endif  // SINGA_UTILS_SAFE_QUEUE_H_
