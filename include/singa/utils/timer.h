/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SINGA_UTILS_TIMER_H
#define SINGA_UTILS_TIMER_H

#include <chrono>

namespace singa {

/// For benchmarking the time cost of operations.
class Timer {
 public:
  typedef std::chrono::duration<int> Seconds;
  typedef std::chrono::duration<int, std::milli> Milliseconds;
  typedef std::chrono::duration<int, std::ratio<60 * 60>> Hours;
  typedef std::chrono::duration<int, std::micro> Microseconds;

  /// Init the internal time point to the current time
  Timer() { Tick(); }
  /// Reset the internal time point to the current time
  void Tick() { last_ = std::chrono::high_resolution_clock::now(); }
  /// Return the duration since last call to Tick() or since the creation of
  /// Timer. The template arg must be from Second or Millisecond or Hour.
  /// The returned value is the count of the time metric.
  template <typename T = Milliseconds>
  int Elapsed() const {
    static_assert(
        std::is_same<T, Seconds>::value ||
            std::is_same<T, Milliseconds>::value ||
            std::is_same<T, Hours>::value ||
            std::is_same<T, Microseconds>::value,
        "Template arg must be Seconds | Milliseconds | Hours | Microseconds");
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<T>(now - last_).count();
  }
  /// Return the string rep of current wall time
  // std::string CurrentTime();

 private:
  std::chrono::high_resolution_clock::time_point last_;
};
}  // namespace singa
#endif
